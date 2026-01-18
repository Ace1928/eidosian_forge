from gitdb.test.lib import (
from gitdb import (
from gitdb.util import hex_to_bin
import zlib
from gitdb.typ import (
import tempfile
import os
from io import BytesIO
def test_decompress_reader(self):
    for close_on_deletion in range(2):
        for with_size in range(2):
            for ds in self.data_sizes:
                cdata = make_bytes(ds, randomize=False)
                if with_size:
                    zdata = zlib.compress(make_object(str_blob_type, cdata))
                    typ, size, reader = DecompressMemMapReader.new(zdata, close_on_deletion)
                    assert size == len(cdata)
                    assert typ == str_blob_type
                    test_reader = DecompressMemMapReader(zdata, close_on_deletion=False)
                    assert test_reader._s == len(cdata)
                else:
                    zdata = zlib.compress(cdata)
                    reader = DecompressMemMapReader(zdata, close_on_deletion, len(cdata))
                    assert reader._s == len(cdata)
                self._assert_stream_reader(reader, cdata, lambda r: r.seek(0))
                dummy = DummyStream()
                reader._m = dummy
                assert not dummy.closed
                del reader
                assert dummy.closed == close_on_deletion