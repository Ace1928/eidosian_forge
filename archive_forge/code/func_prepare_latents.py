import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import PIL
import torch
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from .pipeline_utils import DiffusionPipelineMixin, rescale_noise_cfg
def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, generator=None):
    batch_size = batch_size * num_images_per_prompt
    if image.shape[1] == 4:
        init_latents = image
    else:
        init_latents = self.vae_encoder(sample=image)[0] * self.vae_decoder.config.get('scaling_factor', 0.18215)
    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
        additional_image_per_prompt = batch_size // init_latents.shape[0]
        init_latents = np.concatenate([init_latents] * additional_image_per_prompt, axis=0)
    elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
        raise ValueError(f'Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts.')
    else:
        init_latents = np.concatenate([init_latents], axis=0)
    noise = generator.randn(*init_latents.shape).astype(dtype)
    init_latents = self.scheduler.add_noise(torch.from_numpy(init_latents), torch.from_numpy(noise), torch.from_numpy(timestep))
    return init_latents.numpy()