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
def parse_default_directory_dir(self, filename, parent):
    for d in reversed(xdg_data_dirs):
        self.parse_directory_dir(os.path.join(d, 'desktop-directories'), filename, parent)