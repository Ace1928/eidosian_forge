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
def builtin_command_names():
    """Return list of builtin command names.

    Use of all_command_names() is encouraged rather than builtin_command_names
    and/or plugin_command_names.
    """
    _register_builtin_commands()
    return builtin_command_registry.keys()