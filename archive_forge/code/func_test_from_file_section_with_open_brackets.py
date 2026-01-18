import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_from_file_section_with_open_brackets(self):
    self.assertRaises(ValueError, self.from_file, b'[core\nfoo = bar\n')