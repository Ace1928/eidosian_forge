import zlib
from .. import chunk_writer
from . import TestCaseWithTransport
def check_chunk(self, bytes_list, size):
    data = b''.join(bytes_list)
    self.assertEqual(size, len(data))
    return zlib.decompress(data)