import bz2
import re
from io import BytesIO
import fastbencode as bencode
from .... import errors, iterablefile, lru_cache, multiparent, osutils
from .... import repository as _mod_repository
from .... import revision as _mod_revision
from .... import trace, ui
from ....i18n import ngettext
from ... import pack, serializer
from ... import versionedfile as _mod_versionedfile
from .. import bundle_data
from .. import serializer as bundle_serializer
def iter_diffs(self):
    """Compute the diffs one at a time."""
    self._find_needed_keys()
    needed_ids = [k[-1] for k in self.present_parents]
    needed_ids.extend([k[-1] for k in self.ordered_keys])
    inv_to_lines = self.repo._serializer.write_inventory_to_chunks
    for inv in self.repo.iter_inventories(needed_ids):
        revision_id = inv.revision_id
        key = (revision_id,)
        if key in self.present_parents:
            parent_ids = None
        else:
            parent_ids = [k[-1] for k in self.parent_map[key]]
        as_chunks = inv_to_lines(inv)
        self._process_one_record(key, as_chunks)
        if parent_ids is None:
            continue
        diff = self.diffs.pop(key)
        sha1 = osutils.sha_strings(as_chunks)
        yield (revision_id, parent_ids, sha1, diff)