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
def get_height_width(self, image: [PIL.Image.Image, np.ndarray], height: Optional[int]=None, width: Optional[int]=None):
    """
        This function return the height and width that are downscaled to the next integer multiple of
        `vae_scale_factor`.

        Args:
            image(`PIL.Image.Image`, `np.ndarray`):
                The image input, can be a PIL image, numpy array or pytorch tensor. if it is a numpy array, should have
                shape `[batch, height, width]` or `[batch, height, width, channel]` if it is a pytorch tensor, should
                have shape `[batch, channel, height, width]`.
            height (`int`, *optional*, defaults to `None`):
                The height in preprocessed image. If `None`, will use the height of `image` input.
            width (`int`, *optional*`, defaults to `None`):
                The width in preprocessed. If `None`, will use the width of the `image` input.
        """
    height = height or (image.height if isinstance(image, PIL.Image.Image) else image.shape[-2])
    width = width or (image.width if isinstance(image, PIL.Image.Image) else image.shape[-1])
    width, height = (x - x % self.config.vae_scale_factor for x in (width, height))
    return (height, width)