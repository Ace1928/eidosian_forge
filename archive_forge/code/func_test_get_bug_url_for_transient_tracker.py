from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_get_bug_url_for_transient_tracker(self):
    branch = self.make_branch('some_branch')
    self.assertEqual('http://bugs.example.com/1234', bugtracker.get_bug_url('transient', branch, '1234'))
    self.assertEqual([('get', 'transient', branch), ('get_bug_url', '1234')], self.tracker_type.log)