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
def set_alias(self, alias_name, alias_command):
    """Save the alias in the configuration."""
    with self.lock_write():
        self._set_option(alias_name, alias_command, 'ALIASES')