import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeNullCharacter(self):
    input = '"31337 \\u0000 31337"'
    self.assertEqual(ujson.decode(input), json.loads(input))