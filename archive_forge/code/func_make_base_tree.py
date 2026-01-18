from io import StringIO
from .. import add, errors, tests
from ..bzr import inventory
def make_base_tree(self):
    self.base_tree = self.make_branch_and_tree('base')
    self.build_tree(['base/a', 'base/b', 'base/dir/', 'base/dir/a', 'base/dir/subdir/', 'base/dir/subdir/b'])
    self.base_tree.add(['a', 'b', 'dir', 'dir/a', 'dir/subdir', 'dir/subdir/b'])
    self.base_tree.commit('creating initial tree.')