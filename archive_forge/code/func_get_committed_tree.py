from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def get_committed_tree(self, files, message='Committing'):
    tree = self.get_tree(files)
    tree.add(files)
    tree.commit(message)
    if not tree.has_versioned_directories():
        self.assertInWorkingTree([f for f in files if not f.endswith('/')])
        self.assertPathExists(files)
    else:
        self.assertInWorkingTree(files)
    return tree