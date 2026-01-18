import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
@arch.setter
def arch(self, arch):
    self._arch = arch
    self.set_value('buildozer', 'arch', arch)