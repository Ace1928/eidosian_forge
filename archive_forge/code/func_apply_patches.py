import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def apply_patches(tt, patches, prefix=1):
    """Apply patches to a TreeTransform.

    :param tt: TreeTransform instance
    :param patches: List of patches
    :param prefix: Number leading path segments to strip
    """

    def strip_prefix(p):
        return '/'.join(p.split('/')[1:])
    from breezy.bzr.generate_ids import gen_file_id
    for patch in patches:
        if patch.oldname == b'/dev/null':
            trans_id = None
            orig_contents = b''
        else:
            oldname = strip_prefix(patch.oldname.decode())
            trans_id = tt.trans_id_tree_path(oldname)
            orig_contents = tt._tree.get_file_text(oldname)
            tt.delete_contents(trans_id)
        if patch.newname != b'/dev/null':
            newname = strip_prefix(patch.newname.decode())
            new_contents = iter_patched_from_hunks(orig_contents.splitlines(True), patch.hunks)
            if trans_id is None:
                parts = os.path.split(newname)
                trans_id = tt.root
                for part in parts[1:-1]:
                    trans_id = tt.new_directory(part, trans_id)
                tt.new_file(parts[-1], trans_id, new_contents, file_id=gen_file_id(newname))
            else:
                tt.create_file(new_contents, trans_id)