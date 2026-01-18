import time
from . import debug, errors, osutils, revision, trace
def find_merge_order(self, tip_revision_id, lca_revision_ids):
    """Find the order that each revision was merged into tip.

        This basically just walks backwards with a stack, and walks left-first
        until it finds a node to stop.
        """
    if len(lca_revision_ids) == 1:
        return list(lca_revision_ids)
    looking_for = set(lca_revision_ids)
    stack = [tip_revision_id]
    found = []
    stop = set()
    while stack and looking_for:
        next = stack.pop()
        stop.add(next)
        if next in looking_for:
            found.append(next)
            looking_for.remove(next)
            if len(looking_for) == 1:
                found.append(looking_for.pop())
                break
            continue
        parent_ids = self.get_parent_map([next]).get(next, None)
        if not parent_ids:
            continue
        for parent_id in reversed(parent_ids):
            if parent_id not in stop:
                stack.append(parent_id)
            stop.add(parent_id)
    return found