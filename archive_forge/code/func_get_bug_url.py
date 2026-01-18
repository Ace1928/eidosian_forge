from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def get_bug_url(self, bug_id):
    self.log.append(('get_bug_url', bug_id))
    return 'http://bugs.example.com/%s' % bug_id