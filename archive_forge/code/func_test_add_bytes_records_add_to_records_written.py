from io import BytesIO
from ... import tests
from .. import pack
def test_add_bytes_records_add_to_records_written(self):
    """Adding a Bytes record increments the records_written counter."""
    self.writer.begin()
    self.writer.add_bytes_record([b'foo'], len(b'foo'), names=[])
    self.assertEqual(1, self.writer.records_written)
    self.writer.add_bytes_record([b'foo'], len(b'foo'), names=[])
    self.assertEqual(2, self.writer.records_written)