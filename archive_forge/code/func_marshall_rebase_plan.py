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
def marshall_rebase_plan(last_rev_info, replace_map):
    """Marshall a rebase plan.

    :param last_rev_info: Last revision info tuple.
    :param replace_map: Replace map (old revid -> (new revid, new parents))
    :return: string
    """
    ret = b'# Bazaar rebase plan %d\n' % REBASE_PLAN_VERSION
    ret += b'%d %s\n' % last_rev_info
    for oldrev in replace_map:
        newrev, newparents = replace_map[oldrev]
        ret += b'%s %s' % (oldrev, newrev) + b''.join([b' %s' % p for p in newparents]) + b'\n'
    return ret