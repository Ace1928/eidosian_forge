from unittest import TestCase
import patiencediff
from .. import multiparent, tests
def test_make_patch_from_binary(self):
    patch = multiparent.MultiParent.from_texts(b''.join(LF_SPLIT_LINES))
    expected = multiparent.MultiParent([multiparent.NewText(LF_SPLIT_LINES)])
    self.assertEqual(expected, patch)