from io import BytesIO
from ... import tests
from .. import pack
def test_bytes_record_two_names(self):
    serialiser = pack.ContainerSerialiser()
    record = serialiser.bytes_record(b'bytes', [(b'name1',), (b'name2',)])
    self.assertEqual(b'B5\nname1\nname2\n\nbytes', record)