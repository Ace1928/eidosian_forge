from io import BytesIO
import os
import stat
import sys
from dulwich.diff_tree import (
from dulwich.errors import (
from dulwich.file import GitFile
from dulwich.objects import (
from dulwich.pack import (
from dulwich.refs import ANNOTATED_TAG_SUFFIX
def move_in_pack(self, path):
    """Move a specific file containing a pack into the pack directory.

        Note: The file should be on the same file system as the
            packs directory.

        Args:
          path: Path to the pack file.
        """
    with PackData(path) as p:
        entries = p.sorted_entries()
        basename = self._get_pack_basepath(entries)
        index_name = basename + '.idx'
        if not os.path.exists(index_name):
            with GitFile(index_name, 'wb') as f:
                write_pack_index_v2(f, entries, p.get_stored_checksum())
    for pack in self.packs:
        if pack._basename == basename:
            return pack
    target_pack = basename + '.pack'
    if sys.platform == 'win32':
        try:
            os.remove(target_pack)
        except FileNotFoundError:
            pass
    os.rename(path, target_pack)
    final_pack = Pack(basename)
    self._add_cached_pack(basename, final_pack)
    return final_pack