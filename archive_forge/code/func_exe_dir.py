import configparser
import logging
import warnings
from configparser import ConfigParser
from pathlib import Path
from project import ProjectData
from .commands import run_qmlimportscanner
from . import DEFAULT_APP_ICON
@exe_dir.setter
def exe_dir(self, exe_dir: Path):
    self._exe_dir = exe_dir