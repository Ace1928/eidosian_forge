import time
from . import debug, errors, osutils, revision, trace
def find_distance_to_null(self, target_revision_id, known_revision_ids):
    """Find the left-hand distance to the NULL_REVISION.

        (This can also be considered the revno of a branch at
        target_revision_id.)

        :param target_revision_id: A revision_id which we would like to know
            the revno for.
        :param known_revision_ids: [(revision_id, revno)] A list of known
            revno, revision_id tuples. We'll use this to seed the search.
        """
    known_revnos = dict(known_revision_ids)
    cur_tip = target_revision_id
    num_steps = 0
    NULL_REVISION = revision.NULL_REVISION
    known_revnos[NULL_REVISION] = 0
    searching_known_tips = list(known_revnos)
    unknown_searched = {}
    while cur_tip not in known_revnos:
        unknown_searched[cur_tip] = num_steps
        num_steps += 1
        to_search = {cur_tip}
        to_search.update(searching_known_tips)
        parent_map = self.get_parent_map(to_search)
        parents = parent_map.get(cur_tip, None)
        if not parents:
            raise errors.GhostRevisionsHaveNoRevno(target_revision_id, cur_tip)
        cur_tip = parents[0]
        next_known_tips = []
        for revision_id in searching_known_tips:
            parents = parent_map.get(revision_id, None)
            if not parents:
                continue
            next = parents[0]
            next_revno = known_revnos[revision_id] - 1
            if next in unknown_searched:
                return next_revno + unknown_searched[next]
            if next in known_revnos:
                continue
            known_revnos[next] = next_revno
            next_known_tips.append(next)
        searching_known_tips = next_known_tips
    return known_revnos[cur_tip] + num_steps