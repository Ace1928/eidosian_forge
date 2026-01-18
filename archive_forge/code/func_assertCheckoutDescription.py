import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def assertCheckoutDescription(self, format, expected=None):
    """Assert a checkout's format description matches expectations"""
    if expected is None:
        expected = format
    branch = self.make_branch('%s_cobranch' % format, format=format)
    branch.create_checkout('%s_co' % format, lightweight=True).controldir.destroy_workingtree()
    control = controldir.ControlDir.open('%s_co' % format)
    old_format = control._format.workingtree_format
    try:
        control._format.workingtree_format = controldir.format_registry.make_controldir(format).workingtree_format
        control.create_workingtree()
        tree = workingtree.WorkingTree.open('%s_co' % format)
        format_description = info.describe_format(tree.controldir, tree.branch.repository, tree.branch, tree)
        self.assertEqual(expected, format_description, 'checkout of format called %r was described as %r' % (expected, format_description))
    finally:
        control._format.workingtree_format = old_format