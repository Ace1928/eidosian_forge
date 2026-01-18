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
@classmethod
def from_mergeable(klass, tree, mergeable):
    """Return a Merger for a bundle or merge directive.

        :param tree: The tree to merge changes into
        :param mergeable: A merge directive or bundle
        """
    mergeable.install_revisions(tree.branch.repository)
    base_revision_id, other_revision_id, verified = mergeable.get_merge_request(tree.branch.repository)
    revision_graph = tree.branch.repository.get_graph()
    if base_revision_id is not None:
        if base_revision_id != _mod_revision.NULL_REVISION and revision_graph.is_ancestor(base_revision_id, tree.branch.last_revision()):
            base_revision_id = None
        else:
            trace.warning('Performing cherrypick')
    merger = klass.from_revision_ids(tree, other_revision_id, base_revision_id, revision_graph=revision_graph)
    return (merger, verified)