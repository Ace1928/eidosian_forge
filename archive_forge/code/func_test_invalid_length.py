from io import BytesIO
from ... import tests
from .. import pack
def test_invalid_length(self):
    """If the length-prefix is not a number, parsing raises
        InvalidRecordError.
        """
    parser = self.make_parser_expecting_bytes_record()
    self.assertRaises(pack.InvalidRecordError, parser.accept_bytes, b'not a number\n')