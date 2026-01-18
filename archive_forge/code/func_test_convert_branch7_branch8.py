from .. import branch, controldir, tests, upgrade
from ..bzr import branch as bzrbranch
from ..bzr import workingtree, workingtree_4
def test_convert_branch7_branch8(self):
    b = self.make_branch('branch', format='1.9')
    target = controldir.format_registry.make_controldir('1.9')
    target.set_branch_format(bzrbranch.BzrBranchFormat8())
    converter = b.controldir._format.get_converter(target)
    converter.convert(b.controldir, None)
    b = branch.Branch.open(self.get_url('branch'))
    self.assertIs(b.__class__, bzrbranch.BzrBranch8)
    self.assertEqual({}, b._get_all_reference_info())