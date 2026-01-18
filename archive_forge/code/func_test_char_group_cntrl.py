import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_char_group_cntrl(self):
    self.assertMatchBasenameAndFullpath([('[[:cntrl:]]', ['\x08', '\t', '\x7f'], ['a', 'Q', '茶', '.']), ('[^[:cntrl:]]', ['a', 'Q', '茶', '.'], ['\x08', '\t', '\x7f'])])