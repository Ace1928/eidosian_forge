from io import BytesIO
from .. import errors, filters
from ..filters import (ContentFilter, ContentFilterContext,
from ..osutils import sha_string
from . import TestCase, TestCaseInTempDir
def stk_lookup(key):
    return {'v1': stk1, 'v2': stk2}.get(key)