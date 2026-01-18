import sys
import warnings
import contextlib
import numpy as np
from pathlib import Path
from . import Array, asarray
from .request import ImageMode
from ..config import known_plugins, known_extensions, PluginConfig, FileExtension
from ..config.plugins import _original_order
from .imopen import imopen
def get_format_names(self):
    """Get the names of all registered formats."""
    warnings.warn('`FormatManager` is deprecated and it will be removed in ImageIO v3.To migrate `FormatManager.get_format_names` use `iio.config.known_plugins.keys()` instead.', DeprecationWarning, stacklevel=2)
    return [f.name for f in self._formats]