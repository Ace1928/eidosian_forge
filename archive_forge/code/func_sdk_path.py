import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
@sdk_path.setter
def sdk_path(self, sdk_path: Path):
    self._sdk_path = sdk_path.resolve() if sdk_path else None
    if self._sdk_path:
        self.set_value('buildozer', 'sdk_path', str(self._sdk_path))