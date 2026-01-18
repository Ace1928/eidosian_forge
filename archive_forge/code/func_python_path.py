import configparser
import logging
import warnings
from configparser import ConfigParser
from pathlib import Path
from project import ProjectData
from .commands import run_qmlimportscanner
from . import DEFAULT_APP_ICON
@python_path.setter
def python_path(self, python_path: Path):
    self._python_path = python_path