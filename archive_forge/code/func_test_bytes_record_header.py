from io import BytesIO
from ... import tests
from .. import pack
def test_bytes_record_header(self):
    serialiser = pack.ContainerSerialiser()
    record = serialiser.bytes_header(32, [(b'name1',), (b'name2',)])
    self.assertEqual(b'B32\nname1\nname2\n\n', record)