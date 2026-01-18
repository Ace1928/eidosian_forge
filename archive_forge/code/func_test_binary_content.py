from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_binary_content(self):
    patch = list(multiparent.MultiParent.from_lines(LF_SPLIT_LINES).to_patch())
    multiparent.MultiParent.from_patch(b''.join(patch))