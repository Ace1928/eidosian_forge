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
@staticmethod
def from_revision_ids(tree, other, base=None, other_branch=None, base_branch=None, revision_graph=None, tree_branch=None):
    """Return a Merger for revision-ids.

        :param tree: The tree to merge changes into
        :param other: The revision-id to use as OTHER
        :param base: The revision-id to use as BASE.  If not specified, will
            be auto-selected.
        :param other_branch: A branch containing the other revision-id.  If
            not supplied, tree.branch is used.
        :param base_branch: A branch containing the base revision-id.  If
            not supplied, other_branch or tree.branch will be used.
        :param revision_graph: If you have a revision_graph precomputed, pass
            it in, otherwise it will be created for you.
        :param tree_branch: The branch associated with tree.  If not supplied,
            tree.branch will be used.
        """
    if tree_branch is None:
        tree_branch = tree.branch
    merger = Merger(tree_branch, this_tree=tree, revision_graph=revision_graph)
    if other_branch is None:
        other_branch = tree.branch
    merger.set_other_revision(other, other_branch)
    if base is None:
        merger.find_base()
    else:
        if base_branch is None:
            base_branch = other_branch
        merger.set_base_revision(base, base_branch)
    return merger