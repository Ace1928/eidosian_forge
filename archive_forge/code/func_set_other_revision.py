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
def set_other_revision(self, revision_id, other_branch):
    """Set 'other' based on a branch and revision id

        :param revision_id: The revision to use for a tree
        :param other_branch: The branch containing this tree
        """
    self.other_rev_id = revision_id
    self.other_branch = other_branch
    self._maybe_fetch(other_branch, self.this_branch, self.other_rev_id)
    self.other_tree = self.revision_tree(revision_id)
    self.other_basis = revision_id