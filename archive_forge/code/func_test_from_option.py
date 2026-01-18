from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_from_option(self):
    self.run_bzr('send', retcode=3)
    md = self.get_MD(['--from', 'branch'])
    self.assertEqual(b'rev3', md.revision_id)
    md = self.get_MD(['-f', 'branch'])
    self.assertEqual(b'rev3', md.revision_id)