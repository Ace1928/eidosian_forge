import glob
import os
import sys
import subprocess
import tempfile
import shutil
import atexit
import textwrap
import re
import pytest
import contextlib
import numpy
from pathlib import Path
from numpy.compat import asstr
from numpy._utils import asunicode
from numpy.testing import temppath, IS_WASM
from importlib import import_module
import os
import sys
@_memoize
def build_code(source_code, options=[], skip=[], only=[], suffix=None, module_name=None):
    """
    Compile and import Fortran code using f2py.

    """
    if suffix is None:
        suffix = '.f'
    with temppath(suffix=suffix) as path:
        with open(path, 'w') as f:
            f.write(source_code)
        return build_module([path], options=options, skip=skip, only=only, module_name=module_name)