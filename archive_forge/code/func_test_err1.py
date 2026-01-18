import math
import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util import number
from Cryptodome.Util.number import long_to_bytes
def test_err1(self):
    self.assertRaises(ValueError, long_to_bytes, -1)