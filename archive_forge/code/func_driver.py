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
def driver(self):
    """Low-level HDF5 file driver used to open file"""
    drivers = {h5fd.SEC2: 'sec2', h5fd.STDIO: 'stdio', h5fd.CORE: 'core', h5fd.FAMILY: 'family', h5fd.WINDOWS: 'windows', h5fd.MPIO: 'mpio', h5fd.MPIPOSIX: 'mpiposix', h5fd.fileobj_driver: 'fileobj'}
    if ros3:
        drivers[h5fd.ROS3D] = 'ros3'
    if direct_vfd:
        drivers[h5fd.DIRECT] = 'direct'
    return drivers.get(self.id.get_access_plist().get_driver(), 'unknown')