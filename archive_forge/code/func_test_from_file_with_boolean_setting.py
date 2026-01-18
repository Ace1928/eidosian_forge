import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_from_file_with_boolean_setting(self):
    cf = self.from_file(b'[core]\nfoo\n')
    self.assertEqual(b'true', cf.get((b'core',), b'foo'))