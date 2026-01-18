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
def set_other(self, other_revision, possible_transports=None):
    """Set the revision and tree to merge from.

        This sets the other_tree, other_rev_id, other_basis attributes.

        :param other_revision: The [path, revision] list to merge from.
        """
    self.other_branch, self.other_tree = self._get_tree(other_revision, possible_transports)
    if other_revision[1] == -1:
        self.other_rev_id = self.other_branch.last_revision()
        if _mod_revision.is_null(self.other_rev_id):
            raise errors.NoCommits(self.other_branch)
        self.other_basis = self.other_rev_id
    elif other_revision[1] is not None:
        self.other_rev_id = self.other_branch.get_rev_id(other_revision[1])
        self.other_basis = self.other_rev_id
    else:
        self.other_rev_id = None
        self.other_basis = self.other_branch.last_revision()
        if self.other_basis is None:
            raise errors.NoCommits(self.other_branch)
    if self.other_rev_id is not None:
        self._cached_trees[self.other_rev_id] = self.other_tree
    self._maybe_fetch(self.other_branch, self.this_branch, self.other_basis)