import codecs
from io import BytesIO, StringIO
from .. import annotate, tests
from .ui_testing import StringIOWithEncoding
def assertAnnotateEqualDiff(self, actual, expected):
    if actual != expected:
        self.assertEqualDiff(''.join(('\t'.join(l) for l in expected)), ''.join(('\t'.join(l) for l in actual)))