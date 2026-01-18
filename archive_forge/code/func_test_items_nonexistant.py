import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_items_nonexistant(self):
    cd = ConfigDict()
    cd.set((b'core2',), b'foo', b'bloe')
    self.assertEqual([], list(cd.items((b'core',))))