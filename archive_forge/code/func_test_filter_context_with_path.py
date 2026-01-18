from io import BytesIO
from .. import errors, filters
from ..filters import (ContentFilter, ContentFilterContext,
from ..osutils import sha_string
from . import TestCase, TestCaseInTempDir
def test_filter_context_with_path(self):
    ctx = ContentFilterContext('foo/bar')
    self.assertEqual('foo/bar', ctx.relpath())