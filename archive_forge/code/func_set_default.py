from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
def set_default(self, key):
    """Set the 'default' key to be a clone of the supplied key.

        This method must be called once and only once.
        """
    self.register_alias('default', key)