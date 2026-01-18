from testtools.matchers import Equals, Matcher, Mismatch
from .. import osutils
from .. import revision as _mod_revision
from ..tree import InterTree, TreeChange
def get_tree_layout(self, tree, include_file_ids):
    """Get the (path, file_id) pairs for the current tree."""
    with tree.lock_read():
        for path, ie in tree.iter_entries_by_dir():
            if path != '':
                path += ie.kind_character()
            if include_file_ids:
                yield (path, ie.file_id)
            else:
                yield path