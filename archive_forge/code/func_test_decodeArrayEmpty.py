import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeArrayEmpty(self):
    input = '[]'
    obj = ujson.decode(input)
    self.assertEqual([], obj)