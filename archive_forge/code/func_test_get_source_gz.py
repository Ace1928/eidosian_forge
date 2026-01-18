import gzip
import os
import tempfile
from .... import tests
from ..exporter import (_get_output_stream, check_ref_format,
from . import FastimportFeature
def test_get_source_gz(self):
    fd, filename = tempfile.mkstemp(suffix='.gz')
    os.close(fd)
    with _get_output_stream(filename) as stream:
        stream.write(b'bla')
    with gzip.GzipFile(filename) as f:
        self.assertEqual(b'bla', f.read())