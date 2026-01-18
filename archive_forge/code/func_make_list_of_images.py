import base64
import os
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from packaging import version
from .utils import (
from .utils.constants import (  # noqa: F401
def make_list_of_images(images, expected_ndims: int=3) -> List[ImageInput]:
    """
    Ensure that the input is a list of images. If the input is a single image, it is converted to a list of length 1.
    If the input is a batch of images, it is converted to a list of images.

    Args:
        images (`ImageInput`):
            Image of images to turn into a list of images.
        expected_ndims (`int`, *optional*, defaults to 3):
            Expected number of dimensions for a single input image. If the input image has a different number of
            dimensions, an error is raised.
    """
    if is_batched(images):
        return images
    if isinstance(images, PIL.Image.Image):
        return [images]
    if is_valid_image(images):
        if images.ndim == expected_ndims + 1:
            images = list(images)
        elif images.ndim == expected_ndims:
            images = [images]
        else:
            raise ValueError(f'Invalid image shape. Expected either {expected_ndims + 1} or {expected_ndims} dimensions, but got {images.ndim} dimensions.')
        return images
    raise ValueError(f'Invalid image type. Expected either PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray, but got {type(images)}.')