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
def iter_data(self):
    """iter_data()

            Iterate over all images in the series. (Note: you can also
            iterate over the reader object.)

            """
    self._checkClosed()
    n = self.get_length()
    i = 0
    while i < n:
        try:
            im, meta = self._get_data(i)
        except StopIteration:
            return
        except IndexError:
            if n == float('inf'):
                return
            raise
        yield Array(im, meta)
        i += 1