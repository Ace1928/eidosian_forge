import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_comment_character_within_section_string(self):
    cf = self.from_file(b'[branch "foo#bar"] # a comment\nbar= foo\n')
    self.assertEqual(ConfigFile({(b'branch', b'foo#bar'): {b'bar': b'foo'}}), cf)