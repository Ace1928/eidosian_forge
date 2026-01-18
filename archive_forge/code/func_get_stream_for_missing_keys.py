import time
from .. import controldir, debug, errors, osutils
from .. import revision as _mod_revision
from .. import trace, ui
from ..bzr import chk_map, chk_serializer
from ..bzr import index as _mod_index
from ..bzr import inventory, pack, versionedfile
from ..bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from ..bzr.groupcompress import GroupCompressVersionedFiles, _GCGraphIndex
from ..bzr.vf_repository import StreamSource
from .pack_repo import (NewPack, Pack, PackCommitBuilder, Packer,
from .static_tuple import StaticTuple
def get_stream_for_missing_keys(self, missing_keys):
    missing_inventory_keys = set()
    for key in missing_keys:
        if key[0] != 'inventories':
            raise AssertionError('The only missing keys we should be filling in are inventory keys, not %s' % (key[0],))
        missing_inventory_keys.add(key[1:])
    if self._chk_id_roots or self._chk_p_id_roots:
        raise AssertionError('Cannot call get_stream_for_missing_keys until all of get_stream() has been consumed.')
    yield self._get_inventory_stream(missing_inventory_keys, allow_absent=True)
    yield from self._get_filtered_chk_streams(set())