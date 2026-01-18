import itertools
from .. import debug, revision, trace
from ..graph import DictParentsProvider, Graph, invert_parent_map
from ..repository import AbstractSearchResult
def refine(self, seen, referenced):
    heads = set(self._repo.all_revision_ids())
    heads.difference_update(seen)
    heads.update(referenced)
    return PendingAncestryResult(heads, self._repo)