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
def plugin_name(self):
    """Get the name of the plugin that provides this command.

        :return: The name of the plugin or None if the command is builtin.
        """
    return plugin_name(self.__module__)