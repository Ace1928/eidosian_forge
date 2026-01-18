from io import BytesIO
from ... import tests
from .. import pack
def test_container_with_one_unnamed_record(self):
    """Read a container with one Bytes record.

        Parsing Bytes records is more thoroughly exercised by
        TestBytesRecordReader.  This test is here to ensure that
        ContainerReader's integration with BytesRecordReader is working.
        """
    reader = self.get_reader_for(b'Bazaar pack format 1 (introduced in 0.18)\nB5\n\naaaaaE')
    expected_records = [([], b'aaaaa')]
    self.assertEqual(expected_records, [(names, read_bytes(None)) for names, read_bytes in reader.iter_records()])