import warnings
from typing import List, Optional, Union
import numpy as np
import PIL
import torch
from diffusers import ConfigMixin
from diffusers.image_processor import VaeImageProcessor as DiffusersVaeImageProcessor
from diffusers.utils.pil_utils import PIL_INTERPOLATION
from PIL import Image
from tqdm.auto import tqdm
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = np.std(noise_pred_text, axis=tuple(range(1, noise_pred_text.ndim)), keepdims=True)
    std_cfg = np.std(noise_cfg, axis=tuple(range(1, noise_cfg.ndim)), keepdims=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg