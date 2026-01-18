import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeUnicodeSurrogatePair(self):
    input = 'ð\x90\x8d\x86'
    enc = ujson.encode(input)
    dec = ujson.decode(enc)
    self.assertEqual(enc, json_unicode(input))
    self.assertEqual(dec, json.loads(enc))