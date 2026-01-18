from ... import tests
from ...transport import memory
def test_cat_directory(self):
    wt = self.make_branch_and_tree('a')
    self.build_tree(['a/README'])
    wt.add('README')
    wt.commit('Making sure there is a basis_tree available')
    out, err = self.run_bzr(['cat', '--directory=a', 'README'])
    self.assertEqual('contents of a/README\n', out)