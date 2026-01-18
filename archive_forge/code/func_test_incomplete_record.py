from io import BytesIO
from ... import tests
from .. import pack
def test_incomplete_record(self):
    """If the bytes seen so far don't form a complete record, then there
        will be nothing returned by read_pending_records.
        """
    parser = self.make_parser_expecting_bytes_record()
    parser.accept_bytes(b'5\n\nabcd')
    self.assertEqual([], parser.read_pending_records())