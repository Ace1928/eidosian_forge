from collections import namedtuple
import os
import re
import subprocess
from textwrap import dedent
from numba import config
def check_numpy_version(self):
    cmd = 'python from __future__ import print_function; import types; import numpy;print(numpy.__version__)'
    return self._run_cmd((cmd,))