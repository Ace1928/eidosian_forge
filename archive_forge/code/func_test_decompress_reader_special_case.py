from gitdb.test.lib import (
from gitdb import (
from gitdb.util import hex_to_bin
import zlib
from gitdb.typ import (
import tempfile
import os
from io import BytesIO
def test_decompress_reader_special_case(self):
    odb = LooseObjectDB(fixture_path('objects'))
    mdb = MemoryDB()
    for sha in (b'888401851f15db0eed60eb1bc29dec5ddcace911', b'7bb839852ed5e3a069966281bb08d50012fb309b'):
        ostream = odb.stream(hex_to_bin(sha))
        data = ostream.read()
        assert len(data) == ostream.size
        dump = mdb.store(IStream(ostream.type, ostream.size, BytesIO(data)))
        assert dump.hexsha == sha