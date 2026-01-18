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
def generate_transpose_plan(ancestry, renames, graph, generate_revid):
    """Create a rebase plan that replaces a bunch of revisions
    in a revision graph.

    :param ancestry: Ancestry to consider
    :param renames: Renames of revision
    :param graph: Graph object
    :param generate_revid: Function for creating new revision ids
    """
    replace_map = {}
    todo = []
    children = {}
    parent_map = {}
    for r, ps in ancestry:
        if r not in children:
            children[r] = []
        if ps is None:
            continue
        parent_map[r] = ps
        if r not in children:
            children[r] = []
        for p in ps:
            if p not in children:
                children[p] = []
            children[p].append(r)
    parent_map.update(graph.get_parent_map(filter(lambda x: x not in parent_map, renames.values())))
    for r, v in renames.items():
        replace_map[r] = (v, parent_map[v])
        todo.append(r)
    total = len(todo)
    processed = set()
    i = 0
    pb = ui.ui_factory.nested_progress_bar()
    try:
        while len(todo) > 0:
            r = todo.pop()
            processed.add(r)
            i += 1
            pb.update('determining dependencies', i, total)
            for c in children[r]:
                if c in renames:
                    continue
                if c in replace_map:
                    parents = replace_map[c][1]
                else:
                    parents = parent_map[c]
                assert isinstance(parents, tuple), 'Expected tuple of parents, got: %r' % parents
                if not replace_map[r][0] in parents:
                    parents = list(parents)
                    parents[parents.index(r)] = replace_map[r][0]
                    parents = tuple(parents)
                replace_map[c] = (generate_revid(c, tuple(parents)), tuple(parents))
                if replace_map[c][0] == c:
                    del replace_map[c]
                elif c not in processed:
                    todo.append(c)
    finally:
        pb.finished()
    for revid in renames:
        if revid in replace_map:
            del replace_map[revid]
    return replace_map