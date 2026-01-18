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
def find_base(self):
    revisions = [self.this_basis, self.other_basis]
    if _mod_revision.NULL_REVISION in revisions:
        self.base_rev_id = _mod_revision.NULL_REVISION
        self.base_tree = self.revision_tree(self.base_rev_id)
        self._is_criss_cross = False
    else:
        lcas = self.revision_graph.find_lca(revisions[0], revisions[1])
        self._is_criss_cross = False
        if len(lcas) == 0:
            self.base_rev_id = _mod_revision.NULL_REVISION
        elif len(lcas) == 1:
            self.base_rev_id = list(lcas)[0]
        else:
            self._is_criss_cross = True
            if len(lcas) > 2:
                self.base_rev_id = self.revision_graph.find_unique_lca(revisions[0], revisions[1])
            else:
                self.base_rev_id = self.revision_graph.find_unique_lca(*lcas)
            sorted_lca_keys = self.revision_graph.find_merge_order(revisions[0], lcas)
            if self.base_rev_id == _mod_revision.NULL_REVISION:
                self.base_rev_id = sorted_lca_keys[0]
        if self.base_rev_id == _mod_revision.NULL_REVISION:
            raise errors.UnrelatedBranches()
        if self._is_criss_cross:
            trace.warning('Warning: criss-cross merge encountered.  See bzr help criss-cross.')
            trace.mutter('Criss-cross lcas: %r' % lcas)
            if self.base_rev_id in lcas:
                trace.mutter('Unable to find unique lca. Fallback %r as best option.' % self.base_rev_id)
            interesting_revision_ids = set(lcas)
            interesting_revision_ids.add(self.base_rev_id)
            interesting_trees = {t.get_revision_id(): t for t in self.this_branch.repository.revision_trees(interesting_revision_ids)}
            self._cached_trees.update(interesting_trees)
            if self.base_rev_id in lcas:
                self.base_tree = interesting_trees[self.base_rev_id]
            else:
                self.base_tree = interesting_trees.pop(self.base_rev_id)
            self._lca_trees = [interesting_trees[key] for key in sorted_lca_keys]
        else:
            self.base_tree = self.revision_tree(self.base_rev_id)
    self.base_is_ancestor = True
    self.base_is_other_ancestor = True
    trace.mutter('Base revid: %r' % self.base_rev_id)