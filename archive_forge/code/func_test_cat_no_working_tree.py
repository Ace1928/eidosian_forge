from ... import tests
from ...transport import memory
def test_cat_no_working_tree(self):
    wt = self.make_branch_and_tree('.')
    self.build_tree(['README'])
    wt.add('README')
    wt.commit('Making sure there is a basis_tree available')
    wt.branch.controldir.destroy_workingtree()
    url = self.get_readonly_url() + '/README'
    out, err = self.run_bzr(['cat', url])
    self.assertEqual('contents of README\n', out)