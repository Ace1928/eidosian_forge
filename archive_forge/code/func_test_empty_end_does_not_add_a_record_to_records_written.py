from io import BytesIO
from ... import tests
from .. import pack
def test_empty_end_does_not_add_a_record_to_records_written(self):
    """The end() method does not count towards the records written."""
    self.writer.begin()
    self.writer.end()
    self.assertEqual(0, self.writer.records_written)