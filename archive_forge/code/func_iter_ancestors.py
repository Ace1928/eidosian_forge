from typing import Dict, List, Optional, Tuple
from . import errors, osutils
def iter_ancestors(revision_id: RevisionID, revision_source, only_present: bool=False):
    ancestors = [revision_id]
    distance = 0
    while len(ancestors) > 0:
        new_ancestors: List[bytes] = []
        for ancestor in ancestors:
            if not only_present:
                yield (ancestor, distance)
            try:
                revision = revision_source.get_revision(ancestor)
            except errors.NoSuchRevision as e:
                if e.revision == revision_id:
                    raise
                else:
                    continue
            if only_present:
                yield (ancestor, distance)
            new_ancestors.extend(revision.parent_ids)
        ancestors = new_ancestors
        distance += 1