import os
import sys
import threading
import warnings
from . import process
from .exceptions import (  # noqa
def forking_enable(self, value):
    if not value:
        from ._ext import supports_exec
        if supports_exec:
            self.set_start_method('spawn', force=True)
        else:
            warnings.warn(RuntimeWarning(W_NO_EXECV))