import functools
import inspect
import math
from numbers import Number, Real
import textwrap
from types import SimpleNamespace
from collections import namedtuple
from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib as mpl
from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
from .bezier import (
from .path import Path
from ._enums import JoinStyle, CapStyle
def set_patchB(self, patchB):
    """
        Set the head patch.

        Parameters
        ----------
        patchB : `.patches.Patch`
        """
    self.patchB = patchB
    self.stale = True