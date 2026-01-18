import sys
import os
from warnings import warn
from .compat import filename_decode, filename_encode
from .base import phil, with_phil
from .group import Group
from .. import h5, h5f, h5p, h5i, h5fd, _objects
from .. import version
def registered_drivers():
    """Return a frozenset of the names of all of the registered drivers.
    """
    return frozenset(_drivers)