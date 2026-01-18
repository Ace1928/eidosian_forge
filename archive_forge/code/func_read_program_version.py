import os
import os.path
import warnings
from ..base import CommandLine
def read_program_version(s):
    if 'program' in s:
        return s.split(':')[1].strip()
    return None