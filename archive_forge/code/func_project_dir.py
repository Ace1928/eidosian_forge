import configparser
import logging
import warnings
from configparser import ConfigParser
from pathlib import Path
from project import ProjectData
from .commands import run_qmlimportscanner
from . import DEFAULT_APP_ICON
@project_dir.setter
def project_dir(self, project_dir):
    self._project_dir = project_dir