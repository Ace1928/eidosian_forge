import re
import os
import sys
import warnings
from dill import _dill, Pickler, Unpickler
from ._dill import (
from typing import Optional, Union
import pathlib
import tempfile
def load_session(filename=None, main=None, **kwds):
    warnings.warn('load_session() has been renamed load_module().', PendingDeprecationWarning)
    load_module(filename, module=main, **kwds)