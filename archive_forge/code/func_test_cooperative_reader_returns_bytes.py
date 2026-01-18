import tempfile
from oslotest import base
from glance_store.common import utils
def test_cooperative_reader_returns_bytes(self):
    with tempfile.TemporaryFile() as fd:
        reader = utils.CooperativeReader(fd)
        reader.read = utils.CooperativeReader.read
        out = reader.read(reader)
        self.assertEqual(out, b'')