import gzip
import os
import tempfile
from .... import tests
from ..exporter import (_get_output_stream, check_ref_format,
from . import FastimportFeature
def test_get_output_stream_stdout(self):
    self.assertIsNot(None, _get_output_stream('-'))