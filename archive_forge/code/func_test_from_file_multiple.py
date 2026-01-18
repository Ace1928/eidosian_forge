import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_from_file_multiple(self):
    cf = self.from_file(b'[core]\nfoo = bar\nfoo = blah\n')
    self.assertEqual([b'bar', b'blah'], list(cf.get_multivar((b'core',), b'foo')))
    self.assertEqual([], list(cf.get_multivar((b'core',), b'blah')))