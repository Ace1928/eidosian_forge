from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
def set_default_repository(self, key):
    """Set the FormatRegistry default and Repository default.

        This is a transitional method while Repository.set_default_format
        is deprecated.
        """
    if 'default' in self:
        self.remove('default')
    self.set_default(key)
    format = self.get('default')()