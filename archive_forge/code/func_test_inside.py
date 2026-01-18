import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def test_inside(self):
    self.assertAccess('<asd|>')
    self.assertAccess('<asd|fg>')