import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeListLongConversion(self):
    input = [9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807]
    output = ujson.encode(input)
    self.assertEqual(input, json.loads(output))
    self.assertEqual(input, ujson.decode(output))