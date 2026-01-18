from io import BytesIO
from ... import tests
from .. import pack
def test_bytes_record_one_name_with_two_parts(self):
    serialiser = pack.ContainerSerialiser()
    record = serialiser.bytes_record(b'bytes', [(b'part1', b'part2')])
    self.assertEqual(b'B5\npart1\x00part2\n\nbytes', record)