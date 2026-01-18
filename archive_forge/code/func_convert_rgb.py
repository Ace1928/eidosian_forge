import base64
import os
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from packaging import version
from .utils import (
from .utils.constants import (  # noqa: F401
def convert_rgb(self, image):
    """
        Converts `PIL.Image.Image` to RGB format.

        Args:
            image (`PIL.Image.Image`):
                The image to convert.
        """
    self._ensure_format_supported(image)
    if not isinstance(image, PIL.Image.Image):
        return image
    return image.convert('RGB')