import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
@ndk_path.setter
def ndk_path(self, ndk_path: Path):
    self._ndk_path = ndk_path.resolve() if ndk_path else None
    if self._ndk_path:
        self.set_value('buildozer', 'ndk_path', str(self._ndk_path))