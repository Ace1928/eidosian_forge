from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_message_option(self):
    self.run_bzr('send', retcode=3)
    md = self.get_MD([])
    self.assertIs(None, md.message)
    md = self.get_MD(['-m', 'my message'])
    self.assertEqual('my message', md.message)