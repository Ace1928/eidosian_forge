import os
import sys
from IPython.core.magic import Magics, magics_class, line_magic
from warnings import warn
from traitlets import Bool
@line_magic
def logstate(self, parameter_s=''):
    """Print the status of the logging system."""
    self.shell.logger.logstate()