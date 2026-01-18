from breezy import pyutils, transport
from breezy.bzr.vf_repository import InterDifferingSerializer
from breezy.errors import UninitializableFormat
from breezy.repository import InterRepository, format_registry
from breezy.tests import TestSkipped, default_transport, multiply_tests
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.transport import FileExists
def make_to_repository(self, relpath):
    made_control = self.make_controldir(relpath, self.repository_format_to._matchingcontroldir)
    return self.repository_format_to.initialize(made_control)