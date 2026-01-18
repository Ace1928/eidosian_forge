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
@staticmethod
def numpy_to_pil(images):
    """
        Converts a numpy image or a batch of images to a PIL image.
        """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype('uint8')
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode='L') for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images