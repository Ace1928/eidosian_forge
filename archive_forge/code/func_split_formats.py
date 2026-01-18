from breezy import tests, workingtree
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack4
from breezy.bzr.knitrepo import RepositoryFormatKnit4
def split_formats(self, format, repo_format):
    tree = self.make_branch_and_tree('rich-root', format=format)
    self.build_tree(['rich-root/a/'])
    tree.add('a')
    self.run_bzr(['split', 'rich-root/a'])
    subtree = workingtree.WorkingTree.open('rich-root/a')
    self.assertIsInstance(subtree.branch.repository._format, repo_format)