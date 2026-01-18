from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
def make_branch_supporting_tags(self, relpath):
    return self.make_branch(relpath, format='dirstate-tags')