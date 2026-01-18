import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
@jars_dir.setter
def jars_dir(self, jars_dir: Path):
    self._jars_dir = jars_dir.resolve() if jars_dir else None
    if self._jars_dir:
        self.set_value('buildozer', 'jars_dir', str(self._jars_dir))