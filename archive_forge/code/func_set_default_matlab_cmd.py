import os
from .. import config
from .base import (
@classmethod
def set_default_matlab_cmd(cls, matlab_cmd):
    """Set the default MATLAB command line for MATLAB classes.

        This method is used to set values for all MATLAB
        subclasses.  However, setting this will not update the output
        type for any existing instances.  For these, assign the
        <instance>.inputs.matlab_cmd.
        """
    cls._default_matlab_cmd = matlab_cmd