import base64
import os
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from packaging import version
from .utils import (
from .utils.constants import (  # noqa: F401
def infer_channel_dimension_format(image: np.ndarray, num_channels: Optional[Union[int, Tuple[int, ...]]]=None) -> ChannelDimension:
    """
    Infers the channel dimension format of `image`.

    Args:
        image (`np.ndarray`):
            The image to infer the channel dimension of.
        num_channels (`int` or `Tuple[int, ...]`, *optional*, defaults to `(1, 3)`):
            The number of channels of the image.

    Returns:
        The channel dimension of the image.
    """
    num_channels = num_channels if num_channels is not None else (1, 3)
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels
    if image.ndim == 3:
        first_dim, last_dim = (0, 2)
    elif image.ndim == 4:
        first_dim, last_dim = (1, 3)
    else:
        raise ValueError(f'Unsupported number of image dimensions: {image.ndim}')
    if image.shape[first_dim] in num_channels:
        return ChannelDimension.FIRST
    elif image.shape[last_dim] in num_channels:
        return ChannelDimension.LAST
    raise ValueError('Unable to infer channel dimension format')