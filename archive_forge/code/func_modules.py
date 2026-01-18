import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
@modules.setter
def modules(self, modules):
    self._modules = modules
    self.set_value('buildozer', 'modules', ','.join(modules))