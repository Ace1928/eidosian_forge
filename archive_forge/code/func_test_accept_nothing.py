from io import BytesIO
from ... import tests
from .. import pack
def test_accept_nothing(self):
    """The edge case of parsing an empty string causes no error."""
    parser = self.make_parser_expecting_bytes_record()
    parser.accept_bytes(b'')