import zlib
from .. import chunk_writer
from . import TestCaseWithTransport
def test_chunk_writer_empty(self):
    writer = chunk_writer.ChunkWriter(4096)
    bytes_list, unused, padding = writer.finish()
    node_bytes = self.check_chunk(bytes_list, 4096)
    self.assertEqual(b'', node_bytes)
    self.assertEqual(None, unused)
    self.assertEqual(4088, padding)