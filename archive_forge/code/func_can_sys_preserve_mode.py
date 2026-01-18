import os
import stat
import sys
from .. import atomicfile, osutils
from . import TestCaseInTempDir, TestSkipped
def can_sys_preserve_mode(self):
    return sys.platform not in ('win32',)