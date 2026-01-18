import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
@qt_plugins.setter
def qt_plugins(self, qt_plugins):
    self._qt_plugins = qt_plugins
    self.set_value('android', 'plugins', ','.join(qt_plugins))