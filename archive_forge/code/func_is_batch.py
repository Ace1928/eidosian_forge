import re
import warnings
from numbers import Number
from pathlib import Path
from typing import Dict
import numpy as np
from imageio.core.legacy_plugin_wrapper import LegacyPlugin
from imageio.core.util import Array
from imageio.core.v3_plugin_api import PluginV3
from . import formats
from .config import known_extensions, known_plugins
from .core import RETURN_BYTES
from .core.imopen import imopen
def is_batch(ndimage):
    if isinstance(ndimage, (list, tuple)):
        return True
    ndimage = np.asarray(ndimage)
    if ndimage.ndim <= 2:
        return False
    elif ndimage.ndim == 3 and ndimage.shape[2] < 5:
        return False
    return True