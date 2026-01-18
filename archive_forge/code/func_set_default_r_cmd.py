import os
from shutil import which
from .. import config
from .base import (
def set_default_r_cmd(self, r_cmd):
    """Set the default R command line for R classes.

        This method is used to set values for all R
        subclasses.
        """
    self._cmd = r_cmd