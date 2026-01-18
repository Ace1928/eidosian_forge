import configparser
import logging
import warnings
from configparser import ConfigParser
from pathlib import Path
from project import ProjectData
from .commands import run_qmlimportscanner
from . import DEFAULT_APP_ICON
@source_file.setter
def source_file(self, source_file: Path):
    self._source_file = source_file