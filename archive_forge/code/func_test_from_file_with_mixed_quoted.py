import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_from_file_with_mixed_quoted(self):
    cf = self.from_file(b'[core]\nfoo = "bar"la\n')
    self.assertEqual(b'barla', cf.get((b'core',), b'foo'))