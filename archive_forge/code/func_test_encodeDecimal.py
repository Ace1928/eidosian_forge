import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeDecimal(self):
    sut = decimal.Decimal('1337.1337')
    encoded = ujson.encode(sut)
    decoded = ujson.decode(encoded)
    self.assertEqual(decoded, 1337.1337)