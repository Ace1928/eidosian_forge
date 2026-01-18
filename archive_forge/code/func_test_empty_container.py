from io import BytesIO
from ... import tests
from .. import pack
def test_empty_container(self):
    """Read an empty container."""
    reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nE')
    self.assertEqual([], list(reader.iter_records()))