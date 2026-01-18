from io import BytesIO
from .. import errors, filters
from ..filters import (ContentFilter, ContentFilterContext,
from ..osutils import sha_string
from . import TestCase, TestCaseInTempDir
def test_filtered_output_bytes(self):
    self.assertEqual(_sample_external, list(filtered_output_bytes(_sample_external, None)))
    self.assertEqual(_sample_external, list(filtered_output_bytes(_internal_1, _stack_1)))
    self.assertEqual(_sample_external, list(filtered_output_bytes(_internal_2, _stack_2)))