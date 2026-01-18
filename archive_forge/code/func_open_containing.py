from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
@classmethod
def open_containing(klass, url, possible_transports=None):
    """Open an existing branch which contains url.

        Args:
          url: url to search from.

        See open_containing_from_transport for more detail.
        """
    transport = _mod_transport.get_transport(url, possible_transports)
    return klass.open_containing_from_transport(transport)