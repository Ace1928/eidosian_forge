import os
import os.path
import warnings
from ..base import CommandLine
def read_netcdf_version(s):
    if 'netcdf' in s:
        return ' '.join(s.split(':')[1:]).strip()
    return None