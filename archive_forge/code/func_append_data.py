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
def append_data(self, im, meta=None):
    """append_data(im, meta={})

            Append an image (and meta data) to the file. The final meta
            data that is used consists of the meta data on the given
            image (if applicable), updated with the given meta data.
            """
    self._checkClosed()
    if not isinstance(im, np.ndarray):
        raise ValueError('append_data requires ndarray as first arg')
    total_meta = {}
    if hasattr(im, 'meta') and isinstance(im.meta, dict):
        total_meta.update(im.meta)
    if meta is None:
        pass
    elif not isinstance(meta, dict):
        raise ValueError('Meta must be a dict.')
    else:
        total_meta.update(meta)
    im = asarray(im)
    return self._append_data(im, total_meta)