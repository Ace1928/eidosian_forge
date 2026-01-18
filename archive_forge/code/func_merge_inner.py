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
def merge_inner(this_branch, other_tree, base_tree, ignore_zero=False, backup_files=False, merge_type=Merge3Merger, show_base=False, reprocess=False, other_rev_id=None, interesting_files=None, this_tree=None, change_reporter=None):
    """Primary interface for merging.

    Typical use is probably::

        merge_inner(branch, branch.get_revision_tree(other_revision),
                    branch.get_revision_tree(base_revision))
    """
    if this_tree is None:
        raise errors.BzrError('breezy.merge.merge_inner requires a this_tree parameter')
    merger = Merger(this_branch, other_tree, base_tree, this_tree=this_tree, change_reporter=change_reporter)
    merger.backup_files = backup_files
    merger.merge_type = merge_type
    merger.ignore_zero = ignore_zero
    merger.interesting_files = interesting_files
    merger.show_base = show_base
    merger.reprocess = reprocess
    merger.other_rev_id = other_rev_id
    merger.other_basis = other_rev_id
    get_revision_id = getattr(base_tree, 'get_revision_id', None)
    if get_revision_id is None:
        get_revision_id = base_tree.last_revision
    merger.cache_trees_with_revision_ids([other_tree, base_tree, this_tree])
    merger.set_base_revision(get_revision_id(), this_branch)
    return merger.do_merge()