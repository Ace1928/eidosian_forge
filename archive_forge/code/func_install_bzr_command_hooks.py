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
def install_bzr_command_hooks():
    """Install the hooks to supply bzr's own commands."""
    if _list_bzr_commands in Command.hooks['list_commands']:
        return
    Command.hooks.install_named_hook('list_commands', _list_bzr_commands, 'bzr commands')
    Command.hooks.install_named_hook('get_command', _get_bzr_command, 'bzr commands')
    Command.hooks.install_named_hook('get_command', _get_plugin_command, 'bzr plugin commands')
    Command.hooks.install_named_hook('get_command', _get_external_command, 'bzr external command lookup')
    Command.hooks.install_named_hook('get_missing_command', _try_plugin_provider, 'bzr plugin-provider-db check')