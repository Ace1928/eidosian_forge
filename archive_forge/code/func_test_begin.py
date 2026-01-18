from io import BytesIO
from ... import tests
from .. import pack
def test_begin(self):
    """The begin() method writes the container format marker line."""
    self.writer.begin()
    self.assertOutput(b'Bazaar pack format 1 (introduced in 0.18)\n')