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
def parse_kde_legacy_dirs(self, filename, parent):
    try:
        proc = subprocess.Popen(['kde-config', '--path', 'apps'], stdout=subprocess.PIPE, universal_newlines=True)
        output = proc.communicate()[0].splitlines()
    except OSError:
        return
    try:
        for dir_ in output[0].split(':'):
            self.parse_legacy_dir(dir_, 'kde', filename, parent)
    except IndexError:
        pass