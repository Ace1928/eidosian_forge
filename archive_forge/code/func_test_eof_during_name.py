from io import BytesIO
from ... import tests
from .. import pack
def test_eof_during_name(self):
    """EOF during reading a name."""
    reader = self.get_reader_for(b'123\nname')
    self.assertRaises(pack.UnexpectedEndOfContainerError, reader.read)