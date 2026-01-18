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
def get_bzr_remote_path(self):
    try:
        return os.environ['BZR_REMOTE_PATH']
    except KeyError:
        path = self.get_user_option('bzr_remote_path')
        if path is None:
            path = 'bzr'
        return path