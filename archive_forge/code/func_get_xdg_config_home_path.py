import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def get_xdg_config_home_path(*path_segments):
    xdg_config_home = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config/'))
    return os.path.join(xdg_config_home, *path_segments)