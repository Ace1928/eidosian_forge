from .. import tests
from . import features
def test_no_final_newline(self):
    self.assertChunksToLines([b'foo\n', b'bar\r\n', b'ba\rz'], [b'foo\nbar\r\nba\rz'])
    self.assertChunksToLines([b'foo\n', b'bar\r\n', b'ba\rz'], [b'foo\n', b'bar\r\n', b'ba\rz'], alreadly_lines=True)
    self.assertChunksToLines((b'foo\n', b'bar\r\n', b'ba\rz'), (b'foo\n', b'bar\r\n', b'ba\rz'), alreadly_lines=True)
    self.assertChunksToLines([], [], alreadly_lines=True)
    self.assertChunksToLines([b'foobarbaz'], [b'foobarbaz'], alreadly_lines=True)
    self.assertChunksToLines([], [b''])