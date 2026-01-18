import sys
import os
from warnings import warn
from .compat import filename_decode, filename_encode
from .base import phil, with_phil
from .group import Group
from .. import h5, h5f, h5p, h5i, h5fd, _objects
from .. import version
@atomic.setter
@with_phil
def atomic(self, value):
    self.id.set_mpi_atomicity(value)