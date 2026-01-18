from .. import tests
from . import features
def test_lines_to_lines(self):
    self.assertChunksToLines([b'foo\n', b'bar\r\n', b'ba\rz\n'], [b'foo\n', b'bar\r\n', b'ba\rz\n'], alreadly_lines=True)