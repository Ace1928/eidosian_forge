import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
def cache_trees_with_revision_ids(self, trees):
    """Cache any tree in trees if it has a revision_id."""
    for maybe_tree in trees:
        if maybe_tree is None:
            continue
        try:
            rev_id = maybe_tree.get_revision_id()
        except AttributeError:
            continue
        self._cached_trees[rev_id] = maybe_tree