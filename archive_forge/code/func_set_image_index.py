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
def set_image_index(self, index, **kwargs):
    """set_image_index(index)

            Set the internal pointer such that the next call to
            get_next_data() returns the image specified by the index
            """
    self._checkClosed()
    n = self.get_length()
    self._BaseReaderWriter_last_index = min(max(index - 1, -1), n)