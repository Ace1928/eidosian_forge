import inspect
from typing import Callable, List, Optional, Union
import numpy as np
import PIL
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import PIL_INTERPOLATION
from .pipeline_stable_diffusion import StableDiffusionPipelineMixin
def prepare_mask_and_masked_image(image, mask, latents_shape, vae_scale_factor):
    image = np.array(image.convert('RGB').resize((latents_shape[1] * vae_scale_factor, latents_shape[0] * vae_scale_factor)))
    image = image[None].transpose(0, 3, 1, 2)
    image = image.astype(np.float32) / 127.5 - 1.0
    image_mask = np.array(mask.convert('L').resize((latents_shape[1] * vae_scale_factor, latents_shape[0] * vae_scale_factor)))
    masked_image = image * (image_mask < 127.5)
    mask = mask.resize((latents_shape[1], latents_shape[0]), PIL_INTERPOLATION['nearest'])
    mask = np.array(mask.convert('L'))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    return (mask, masked_image)