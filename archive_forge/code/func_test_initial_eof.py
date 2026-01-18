from io import BytesIO
from ... import tests
from .. import pack
def test_initial_eof(self):
    """EOF before any bytes read at all."""
    reader = self.get_reader_for(b'')
    self.assertRaises(pack.UnexpectedEndOfContainerError, reader.read)