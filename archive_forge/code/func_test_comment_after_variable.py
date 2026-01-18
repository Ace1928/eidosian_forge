import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_comment_after_variable(self):
    cf = self.from_file(b'[section]\nbar= foo # a comment\n')
    self.assertEqual(ConfigFile({(b'section',): {b'bar': b'foo'}}), cf)