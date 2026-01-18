import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_loadFile(self):
    f = StringIO('[1,2,3,4]')
    self.assertEqual([1, 2, 3, 4], ujson.load(f))