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
def get_see_also(self, additional_terms=None):
    """Return a list of help topics that are related to this command.

        The list is derived from the content of the _see_also attribute. Any
        duplicates are removed and the result is in lexical order.

        Args:
          additional_terms: Additional help topics to cross-reference.

        Returns:
          A list of help topics.
        """
    see_also = set(getattr(self, '_see_also', []))
    if additional_terms:
        see_also.update(additional_terms)
    return sorted(see_also)