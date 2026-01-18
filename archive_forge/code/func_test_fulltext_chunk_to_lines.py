from .. import tests
from . import features
def test_fulltext_chunk_to_lines(self):
    self.assertChunksToLines([b'foo\n', b'bar\r\n', b'ba\rz\n'], [b'foo\nbar\r\nba\rz\n'])
    self.assertChunksToLines([b'foobarbaz\n'], [b'foobarbaz\n'], alreadly_lines=True)
    self.assertChunksToLines([b'foo\n', b'bar\n', b'\n', b'baz\n', b'\n', b'\n'], [b'foo\nbar\n\nbaz\n\n\n'])
    self.assertChunksToLines([b'foobarbaz'], [b'foobarbaz'], alreadly_lines=True)
    self.assertChunksToLines([b'foobarbaz'], [b'foo', b'bar', b'baz'])