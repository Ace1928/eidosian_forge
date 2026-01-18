import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeControlEscaping(self):
    input = '\x19'
    enc = ujson.encode(input)
    dec = ujson.decode(enc)
    self.assertEqual(input, dec)
    self.assertEqual(enc, json_unicode(input))