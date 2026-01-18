import abc
import base64
import contextlib
from io import BytesIO, TextIOWrapper
import itertools
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import uuid
import warnings
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib._animation_data import (
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
@frame_format.setter
def frame_format(self, frame_format):
    if frame_format in self.supported_formats:
        self._frame_format = frame_format
    else:
        _api.warn_external(f'Ignoring file format {frame_format!r} which is not supported by {type(self).__name__}; using {self.supported_formats[0]} instead.')
        self._frame_format = self.supported_formats[0]