import os
from .. import config
from .base import (
@classmethod
def set_default_paths(cls, paths):
    """Set the default MATLAB paths for MATLAB classes.

        This method is used to set values for all MATLAB
        subclasses.  However, setting this will not update the output
        type for any existing instances.  For these, assign the
        <instance>.inputs.paths.
        """
    cls._default_paths = paths