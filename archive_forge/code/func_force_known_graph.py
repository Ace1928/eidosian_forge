from breezy import pyutils, transport
from breezy.bzr.vf_repository import InterDifferingSerializer
from breezy.errors import UninitializableFormat
from breezy.repository import InterRepository, format_registry
from breezy.tests import TestSkipped, default_transport, multiply_tests
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.transport import FileExists
def force_known_graph(testcase):
    from breezy.bzr.fetch import Inter1and2Helper
    testcase.overrideAttr(Inter1and2Helper, 'known_graph_threshold', -1)