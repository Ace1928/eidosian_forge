from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
def register_alias(self, key, target, hidden=False):
    """Register a format alias.

        Args:
          key: Alias name
          target: Target format
          hidden: Whether the alias is hidden
        """
    info = self.get_info(target)
    registry.Registry.register_alias(self, key, target, ControlDirFormatInfo(native=info.native, deprecated=info.deprecated, hidden=hidden, experimental=info.experimental))