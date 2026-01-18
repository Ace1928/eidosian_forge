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
def get_aliases(self):
    """Return the aliases section."""
    if 'ALIASES' in self._get_parser():
        return self._get_parser()['ALIASES']
    else:
        return {}