from breezy import pyutils, transport
from breezy.bzr.vf_repository import InterDifferingSerializer
from breezy.errors import UninitializableFormat
from breezy.repository import InterRepository, format_registry
from breezy.tests import TestSkipped, default_transport, multiply_tests
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.transport import FileExists
def make_controldir(self, relpath, format=None):
    try:
        url = self.get_url(relpath)
        segments = url.split('/')
        if segments and segments[-1] not in ('', '.'):
            parent = '/'.join(segments[:-1])
            t = transport.get_transport(parent)
            try:
                t.mkdir(segments[-1])
            except FileExists:
                pass
        if format is None:
            format = self.repository_format._matchingcontroldir
        return format.initialize(url)
    except UninitializableFormat:
        raise TestSkipped('Format %s is not initializable.' % format)