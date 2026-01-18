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
def find_merge_tool(self, name):
    from .mergetools import known_merge_tools
    command_line = self.get_user_option('bzr.mergetool.%s' % name, expand=False) or known_merge_tools.get(name, None)
    return command_line