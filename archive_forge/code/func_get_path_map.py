from testtools.matchers import Equals, Matcher, Mismatch
from .. import osutils
from .. import revision as _mod_revision
from ..tree import InterTree, TreeChange
def get_path_map(self, tree):
    """Get the (path, previous_path) pairs for the current tree."""
    previous_intertree = InterTree.get(self.previous_tree, tree)
    with tree.lock_read(), self.previous_tree.lock_read():
        for path, ie in tree.iter_entries_by_dir():
            if tree.supports_rename_tracking():
                previous_path = previous_intertree.find_source_path(path)
            elif self.previous_tree.is_versioned(path):
                previous_path = path
            else:
                previous_path = None
            if previous_path:
                kind = self.previous_tree.kind(previous_path)
                if kind == 'directory':
                    previous_path += '/'
            if path == '':
                yield ('', previous_path)
            else:
                yield (path + ie.kind_character(), previous_path)