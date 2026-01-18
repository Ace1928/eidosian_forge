import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def test_indexing(self):
    self.assertAccess('abc[def].<ghi|>')
    self.assertAccess('abc[def].<|ghi>')
    self.assertAccess('abc[def].<gh|i>')
    self.assertAccess('abc[def].gh |i')
    self.assertAccess('abc[def]|')