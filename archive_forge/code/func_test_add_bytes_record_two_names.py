from io import BytesIO
from ... import tests
from .. import pack
def test_add_bytes_record_two_names(self):
    """Add a bytes record with two names."""
    self.writer.begin()
    offset, length = self.writer.add_bytes_record([b'abc'], len(b'abc'), names=[(b'name1',), (b'name2',)])
    self.assertEqual((42, 19), (offset, length))
    self.assertOutput(b'Bazaar pack format 1 (introduced in 0.18)\nB3\nname1\nname2\n\nabc')