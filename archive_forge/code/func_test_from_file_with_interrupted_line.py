import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_from_file_with_interrupted_line(self):
    cf = self.from_file(b'[core]\nfoo = bar\\\n la\n')
    self.assertEqual(b'barla', cf.get((b'core',), b'foo'))