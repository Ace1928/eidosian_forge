import os
import warnings
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Optional, Tuple
from warnings import warn
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import whoami
from ..models import DDPOStableDiffusionPipeline
from . import BaseTrainer, DDPOConfig
from .utils import PerPromptStatTracker
def calculate_loss(self, latents, timesteps, next_latents, log_probs, advantages, embeds):
    """
        Calculate the loss for a batch of an unpacked sample

        Args:
            latents (torch.Tensor):
                The latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            timesteps (torch.Tensor):
                The timesteps sampled from the diffusion model, shape: [batch_size]
            next_latents (torch.Tensor):
                The next latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            log_probs (torch.Tensor):
                The log probabilities of the latents, shape: [batch_size]
            advantages (torch.Tensor):
                The advantages of the latents, shape: [batch_size]
            embeds (torch.Tensor):
                The embeddings of the prompts, shape: [2*batch_size or batch_size, ...]
                Note: the "or" is because if train_cfg is True, the expectation is that negative prompts are concatenated to the embeds

        Returns:
            loss (torch.Tensor), approx_kl (torch.Tensor), clipfrac (torch.Tensor)
            (all of these are of shape (1,))
        """
    with self.autocast():
        if self.config.train_cfg:
            noise_pred = self.sd_pipeline.unet(torch.cat([latents] * 2), torch.cat([timesteps] * 2), embeds).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.sample_guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred = self.sd_pipeline.unet(latents, timesteps, embeds).sample
        scheduler_step_output = self.sd_pipeline.scheduler_step(noise_pred, timesteps, latents, eta=self.config.sample_eta, prev_sample=next_latents)
        log_prob = scheduler_step_output.log_probs
    advantages = torch.clamp(advantages, -self.config.train_adv_clip_max, self.config.train_adv_clip_max)
    ratio = torch.exp(log_prob - log_probs)
    loss = self.loss(advantages, self.config.train_clip_range, ratio)
    approx_kl = 0.5 * torch.mean((log_prob - log_probs) ** 2)
    clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float())
    return (loss, approx_kl, clipfrac)