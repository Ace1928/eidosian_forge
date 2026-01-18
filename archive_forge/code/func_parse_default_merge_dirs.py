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
def parse_default_merge_dirs(self, child, filename, parent):
    basename = os.path.splitext(os.path.basename(filename))[0]
    for d in reversed(xdg_config_dirs):
        self.parse_merge_dir(os.path.join(d, 'menus', basename + '-merged'), child, filename, parent)