from typing import Dict, List, Optional, Tuple
from . import errors, osutils
def find_present_ancestors(revision_id: RevisionID, revision_source) -> Dict[RevisionID, Tuple[int, int]]:
    """Return the ancestors of a revision present in a branch.

    It's possible that a branch won't have the complete ancestry of
    one of its revisions.
    """
    found_ancestors: Dict[RevisionID, Tuple[int, int]] = {}
    anc_iter = enumerate(iter_ancestors(revision_id, revision_source, only_present=True))
    for anc_order, (anc_id, anc_distance) in anc_iter:
        if anc_id not in found_ancestors:
            found_ancestors[anc_id] = (anc_order, anc_distance)
    return found_ancestors