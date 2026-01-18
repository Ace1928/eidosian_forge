from xdg.Menu import parse, Menu, MenuEntry
import os
import locale
import subprocess
import ast
import sys
from xdg.BaseDirectory import xdg_data_dirs, xdg_config_dirs
from xdg.DesktopEntry import DesktopEntry
from xdg.Exceptions import ParsingError
from xdg.util import PY3
import xdg.Locale
import xdg.Config
def parse_directory_dir(self, value, filename, parent):
    value = _check_file_path(value, filename, TYPE_DIR)
    if value:
        parent.DirectoryDirs.append(value)