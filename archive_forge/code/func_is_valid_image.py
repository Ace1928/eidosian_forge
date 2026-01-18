import base64
import os
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from packaging import version
from .utils import (
from .utils.constants import (  # noqa: F401
def is_valid_image(img):
    return is_vision_available() and isinstance(img, PIL.Image.Image) or isinstance(img, np.ndarray) or is_torch_tensor(img) or is_tf_tensor(img) or is_jax_tensor(img)