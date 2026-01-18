import os
from .. import config
from .base import (
def get_matlab_command():
    """Determine whether Matlab is installed and can be executed."""
    if 'NIPYPE_NO_MATLAB' not in os.environ:
        from nipype.utils.filemanip import which
        return which(os.getenv('MATLABCMD', 'matlab'))