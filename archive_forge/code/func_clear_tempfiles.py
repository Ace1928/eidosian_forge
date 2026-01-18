import os
import time
import tempfile
import logging
import shutil
import weakref
from pyomo.common.dependencies import attempt_import, pyutilib_available
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import TempfileContextError
from pyomo.common.multithread import MultiThreadWrapperWithMain
def clear_tempfiles(self, remove=True):
    """Delete all temporary files and remove all contexts."""
    while self._context_stack:
        self.pop(remove)