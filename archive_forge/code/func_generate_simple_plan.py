import os
from ... import config as _mod_config
from ... import osutils, ui
from ...bzr.generate_ids import gen_revision_id
from ...bzr.inventorytree import InventoryTreeChange
from ...errors import (BzrError, NoCommonAncestor, UnknownFormatError,
from ...graph import FrozenHeadsCache
from ...merge import Merger
from ...revision import NULL_REVISION
from ...trace import mutter
from ...transport import NoSuchFile
from ...tsort import topo_sort
from .maptree import MapTree, map_file_ids
def generate_simple_plan(todo_set, start_revid, stop_revid, onto_revid, graph, generate_revid, skip_full_merged=False):
    """Create a simple rebase plan that replays history based
    on one revision being replayed on top of another.

    :param todo_set: A set of revisions to rebase. Only the revisions
        topologically between stop_revid and start_revid (inclusive) are
        rebased; other revisions are ignored (and references to them are
        preserved).
    :param start_revid: Id of revision at which to start replaying
    :param stop_revid: Id of revision until which to stop replaying
    :param onto_revid: Id of revision on top of which to replay
    :param graph: Graph object
    :param generate_revid: Function for generating new revision ids
    :param skip_full_merged: Skip revisions that merge already merged
                             revisions.

    :return: replace map
    """
    assert start_revid is None or start_revid in todo_set, 'invalid start revid({!r}), todo_set({!r})'.format(start_revid, todo_set)
    assert stop_revid is None or stop_revid in todo_set, 'invalid stop_revid'
    replace_map = {}
    parent_map = graph.get_parent_map(todo_set)
    order = topo_sort(parent_map)
    if stop_revid is None:
        stop_revid = order[-1]
    if start_revid is None:
        lca = graph.find_lca(stop_revid, onto_revid)
        if lca == {NULL_REVISION}:
            raise UnrelatedBranches()
        start_revid = order[0]
    todo = order[order.index(start_revid):order.index(stop_revid) + 1]
    heads_cache = FrozenHeadsCache(graph)
    for oldrevid in todo:
        oldparents = parent_map[oldrevid]
        assert isinstance(oldparents, tuple), 'not tuple: %r' % oldparents
        parents = []
        if heads_cache.heads((oldparents[0], onto_revid)) == {onto_revid}:
            parents.append(onto_revid)
        elif oldparents[0] in replace_map:
            parents.append(replace_map[oldparents[0]][0])
        else:
            parents.append(onto_revid)
            parents.append(oldparents[0])
        if len(oldparents) > 1:
            additional_parents = heads_cache.heads(oldparents[1:])
            for oldparent in oldparents[1:]:
                if oldparent in additional_parents:
                    if heads_cache.heads((oldparent, onto_revid)) == {onto_revid}:
                        pass
                    elif oldparent in replace_map:
                        newparent = replace_map[oldparent][0]
                        if parents[0] == onto_revid:
                            parents[0] = newparent
                        else:
                            parents.append(newparent)
                    else:
                        parents.append(oldparent)
            if len(parents) == 1 and skip_full_merged:
                continue
        parents = tuple(parents)
        newrevid = generate_revid(oldrevid, parents)
        assert newrevid != oldrevid, 'old and newrevid equal (%r)' % newrevid
        assert isinstance(parents, tuple), 'parents not tuple: %r' % parents
        replace_map[oldrevid] = (newrevid, parents)
    return replace_map