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
def open_tree_or_branch(klass, location, name=None):
    """Return the branch and working tree at a location.

        If there is no tree at the location, tree will be None.
        If there is no branch at the location, an exception will be
        raised
        Returns: (tree, branch)
        """
    controldir = klass.open(location)
    return controldir._get_tree_branch(name=name)