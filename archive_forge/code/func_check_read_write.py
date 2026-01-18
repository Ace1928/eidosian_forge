from io import BytesIO
from pprint import pformat
from ... import errors
from ...osutils import sha_string
from ..weave import Weave, WeaveFormatError, WeaveInvalidChecksum
from ..weavefile import read_weave, write_weave
from . import TestCase, TestCaseInTempDir
def check_read_write(self, k):
    """Check the weave k can be written & re-read."""
    from tempfile import TemporaryFile
    tf = TemporaryFile()
    write_weave(k, tf)
    tf.seek(0)
    k2 = read_weave(tf)
    if k != k2:
        tf.seek(0)
        self.log('serialized weave:')
        self.log(tf.read())
        self.log('')
        self.log('parents: %s' % (k._parents == k2._parents))
        self.log('         %r' % k._parents)
        self.log('         %r' % k2._parents)
        self.log('')
        self.fail('read/write check failed')