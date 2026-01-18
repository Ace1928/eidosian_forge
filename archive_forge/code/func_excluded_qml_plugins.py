import configparser
import logging
import warnings
from configparser import ConfigParser
from pathlib import Path
from project import ProjectData
from .commands import run_qmlimportscanner
from . import DEFAULT_APP_ICON
@excluded_qml_plugins.setter
def excluded_qml_plugins(self, excluded_qml_plugins):
    self._excluded_qml_plugins = excluded_qml_plugins