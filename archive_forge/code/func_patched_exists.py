import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def patched_exists(path):
    if path in ('source.pyx', os.path.join('/work/dir', 'source.pyx'), os.path.join('my_working_path', 'source.pyx'), 'file.pyx', 'file1.pyx', 'file2.pyx', 'file3.pyx', 'foo.pyx', 'bar.pyx'):
        return True
    return unpatched_exists(path)