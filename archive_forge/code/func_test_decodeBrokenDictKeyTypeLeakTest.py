import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_decodeBrokenDictKeyTypeLeakTest(self):
    input = '{{1337:""}}'
    for x in range(1000):
        self.assertRaises(ValueError, ujson.decode, input)