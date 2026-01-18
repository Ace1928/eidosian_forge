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
def save_section(sections, order, label, section):
    if len(section) > 0:
        if label in sections:
            sections[label] += '\n' + section
        else:
            order.append(label)
            sections[label] = section