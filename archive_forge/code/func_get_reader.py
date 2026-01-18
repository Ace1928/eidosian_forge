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
def get_reader(self, request):
    """get_reader(request)

        Return a reader object that can be used to read data and info
        from the given file. Users are encouraged to use
        imageio.get_reader() instead.
        """
    select_mode = request.mode[1] if request.mode[1] in 'iIvV' else ''
    if select_mode not in self.modes:
        raise RuntimeError(f'Format {self.name} cannot read in {request.mode.image_mode} mode')
    return self.Reader(self, request)