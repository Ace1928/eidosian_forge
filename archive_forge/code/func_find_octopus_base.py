from heapq import heappop, heappush
from .lru_cache import LRUCache
def find_octopus_base(repo, commit_ids):
    """Find lowest common ancestors of *all* provided commit_ids.

    Args:
      repo: Repository
      commit_ids:  list of commit ids
    Returns:
      list of lowest common ancestor commit_ids
    """
    cmtcache = LRUCache(max_cache=128)
    parents_provider = repo.parents_provider()

    def lookup_stamp(cmtid):
        if cmtid not in cmtcache:
            cmtcache[cmtid] = repo.object_store[cmtid]
        return cmtcache[cmtid].commit_time

    def lookup_parents(cmtid):
        commit = None
        if cmtid in cmtcache:
            commit = cmtcache[cmtid]
        return parents_provider.get_parents(cmtid, commit=commit)
    if not commit_ids:
        return []
    if len(commit_ids) <= 2:
        return find_merge_base(repo, commit_ids)
    lcas = [commit_ids[0]]
    others = commit_ids[1:]
    for cmt in others:
        next_lcas = []
        for ca in lcas:
            res = _find_lcas(lookup_parents, cmt, [ca], lookup_stamp)
            next_lcas.extend(res)
        lcas = next_lcas[:]
    return lcas