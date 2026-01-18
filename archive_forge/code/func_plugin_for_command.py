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
def plugin_for_command(self, cmd_name):
    """Takes a command and returns the information for that plugin

        :return: A dictionary with all the available information
            for the requested plugin
        """
    raise NotImplementedError