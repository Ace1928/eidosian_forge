from io import BytesIO
from ... import tests
from .. import pack
def test_invalid_record(self):
    """Test the formatting of InvalidRecordError."""
    e = pack.InvalidRecordError('xxx')
    self.assertEqual('Invalid record: xxx', str(e))