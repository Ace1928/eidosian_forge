import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
def get_change_editor(self, old_tree, new_tree):
    from breezy import diff
    cmd = self._get_change_editor()
    if cmd is None:
        return None
    cmd = cmd.replace('@old_path', '{old_path}')
    cmd = cmd.replace('@new_path', '{new_path}')
    cmd = cmdline.split(cmd)
    if '{old_path}' not in cmd:
        cmd.extend(['{old_path}', '{new_path}'])
    return diff.DiffFromTool.from_string(cmd, old_tree, new_tree, sys.stdout)