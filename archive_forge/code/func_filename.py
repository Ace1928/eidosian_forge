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
def filename(self):
    """File name on disk"""
    return filename_decode(h5f.get_name(self.id))