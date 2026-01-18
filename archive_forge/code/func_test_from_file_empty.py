import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_from_file_empty(self):
    cf = self.from_file(b'')
    self.assertEqual(ConfigFile(), cf)