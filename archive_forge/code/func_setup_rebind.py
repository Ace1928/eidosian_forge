from breezy import branch, controldir, errors, tests
from breezy.tests import script
def setup_rebind(self, format):
    branch1 = self.make_branch('branch1')
    branch2 = self.make_branch('branch2', format=format)
    branch2.bind(branch1)
    branch2.unbind()