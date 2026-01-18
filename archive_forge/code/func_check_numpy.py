from collections import namedtuple
import os
import re
import subprocess
from textwrap import dedent
from numba import config
def check_numpy(self):
    cmd = 'python from __future__ import print_function; import types; import numpy; print(isinstance(numpy, types.ModuleType))'
    return self._run_cmd((cmd,))