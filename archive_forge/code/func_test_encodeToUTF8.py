import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeToUTF8(self):
    input = b'\xe6\x97\xa5\xd1\x88'
    input = input.decode('utf-8')
    enc = ujson.encode(input, ensure_ascii=False)
    dec = ujson.decode(enc)
    self.assertEqual(enc, json.dumps(input, ensure_ascii=False))
    self.assertEqual(dec, json.loads(enc))