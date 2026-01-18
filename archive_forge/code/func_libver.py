import sys
import os
from warnings import warn
from .compat import filename_decode, filename_encode
from .base import phil, with_phil
from .group import Group
from .. import h5, h5f, h5p, h5i, h5fd, _objects
from .. import version
@property
@with_phil
def libver(self):
    """File format version bounds (2-tuple: low, high)"""
    bounds = self.id.get_access_plist().get_libver_bounds()
    return tuple((libver_dict_r[x] for x in bounds))