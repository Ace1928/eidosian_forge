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
def get_cmd_object(cmd_name: str, plugins_override: bool=True) -> 'Command':
    """Return the command object for a command.

    plugins_override
        If true, plugin commands can override builtins.
    """
    try:
        return _get_cmd_object(cmd_name, plugins_override)
    except KeyError:
        candidate = guess_command(cmd_name)
        if candidate is not None:
            raise errors.CommandError(gettext('unknown command "%s". Perhaps you meant "%s"') % (cmd_name, candidate))
        raise errors.CommandError(gettext('unknown command "%s"') % cmd_name)