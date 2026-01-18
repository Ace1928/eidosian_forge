import contextlib
import os
import sys
from typing import List, Optional, Type, Union
from . import i18n, option, osutils, trace
from .lazy_import import lazy_import
import breezy
from breezy import (
from . import errors, registry
from .hooks import Hooks
from .i18n import gettext
from .plugin import disable_plugins, load_plugins, plugin_name
def all_command_names():
    """Return a set of all command names."""
    names = set()
    for hook in Command.hooks['list_commands']:
        names = hook(names)
        if names is None:
            raise AssertionError('hook %s returned None' % Command.hooks.get_hook_name(hook))
    return names