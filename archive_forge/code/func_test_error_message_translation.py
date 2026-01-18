import io
from .. import errors, i18n, tests, workingtree
def test_error_message_translation(self):
    """do errors get translated?"""
    err = None
    tree = self.make_branch_and_tree('.')
    try:
        workingtree.WorkingTree.open('./foo')
    except errors.NotBranchError as e:
        err = str(e)
    self.assertContainsRe(err, 'zzå{{Not a branch: .*}}')