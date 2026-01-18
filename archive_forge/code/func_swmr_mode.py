import sys
import os
from warnings import warn
from .compat import filename_decode, filename_encode
from .base import phil, with_phil
from .group import Group
from .. import h5, h5f, h5p, h5i, h5fd, _objects
from .. import version
@swmr_mode.setter
@with_phil
def swmr_mode(self, value):
    if value:
        self.id.start_swmr_write()
    else:
        raise ValueError('It is not possible to forcibly switch SWMR mode off.')