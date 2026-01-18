from io import BytesIO
from ... import tests
from .. import pack
def test_add_second_bytes_record_gets_higher_offset(self):
    self.writer.begin()
    self.writer.add_bytes_record([b'a', b'bc'], len(b'abc'), names=[])
    offset, length = self.writer.add_bytes_record([b'abc'], len(b'abc'), names=[])
    self.assertEqual((49, 7), (offset, length))
    self.assertOutput(b'Bazaar pack format 1 (introduced in 0.18)\nB3\n\nabcB3\n\nabc')