import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_show_diff_non_ascii(self):
    message = 'Message with µ'
    body = b'Body with \xb5\n'
    wt = self.make_branch_and_tree('.')
    self.build_tree_contents([('foo', body)])
    wt.add('foo')
    wt.commit(message=message)
    out, err = self.run_bzr('log -p --long')
    self.assertNotEqual('', out)
    self.assertEqual('', err)
    out, err = self.run_bzr('log -p --short')
    self.assertNotEqual('', out)
    self.assertEqual('', err)
    out, err = self.run_bzr('log -p --line')
    self.assertNotEqual('', out)
    self.assertEqual('', err)