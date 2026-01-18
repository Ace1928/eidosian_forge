import weakref
import numpy as np
from .affines import voxel_sizes
from .optpkg import optional_package
from .orientations import aff2axcodes, axcodes2ornt
@property
def figs(self):
    """A tuple of the figure(s) containing the axes"""
    return tuple(self._figs)