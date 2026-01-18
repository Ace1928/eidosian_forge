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
def search_read_format(self, request):
    """search_read_format(request)

        Search a format that can read a file according to the given request.
        Returns None if no appropriate format was found. (used internally)
        """
    try:
        return imopen(request, request.mode.io_mode, legacy_mode=True)._format
    except AttributeError:
        warnings.warn('ImageIO now uses a v3 plugin when reading this format. Please migrate to the v3 API (preferred) or use imageio.v2.', DeprecationWarning, stacklevel=2)
        return None
    except ValueError:
        return None