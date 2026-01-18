import os
import sys
import threading
import warnings
from . import process
from .exceptions import (  # noqa
def forking_is_enabled(self):
    return (self.get_start_method() or 'fork') == 'fork'