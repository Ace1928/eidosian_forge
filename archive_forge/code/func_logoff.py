import os
import sys
from IPython.core.magic import Magics, magics_class, line_magic
from warnings import warn
from traitlets import Bool
@line_magic
def logoff(self, parameter_s=''):
    """Temporarily stop logging.

        You must have previously started logging."""
    self.shell.logger.switch_log(0)