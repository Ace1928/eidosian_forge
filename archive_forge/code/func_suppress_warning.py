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
def suppress_warning(self, warning):
    """Should the warning be suppressed or emitted.

        Args:
          warning: The name of the warning being tested.

        Returns:
          True if the warning should be suppressed, False otherwise.
        """
    warnings = self.get_user_option_as_list('suppress_warnings')
    if warnings is None or warning not in warnings:
        return False
    else:
        return True