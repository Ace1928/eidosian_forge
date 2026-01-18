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
def find_text_key_references(self):
    """Find the text key references within the repository.

        :return: A dictionary mapping text keys ((fileid, revision_id) tuples)
            to whether they were referred to by the inventory of the
            revision_id that they contain. The inventory texts from all present
            revision ids are assessed to generate this report.
        """
    revision_keys = self.revisions.keys()
    result = {}
    rich_roots = self.supports_rich_root()
    with ui.ui_factory.nested_progress_bar() as pb:
        all_revs = self.all_revision_ids()
        total = len(all_revs)
        for pos, inv in enumerate(self.iter_inventories(all_revs)):
            pb.update('Finding text references', pos, total)
            for _, entry in inv.iter_entries():
                if not rich_roots and entry.file_id == inv.root_id:
                    continue
                key = (entry.file_id, entry.revision)
                result.setdefault(key, False)
                if entry.revision == inv.revision_id:
                    result[key] = True
        return result