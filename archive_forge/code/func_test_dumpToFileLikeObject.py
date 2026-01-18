import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_dumpToFileLikeObject(self):

    class filelike:

        def __init__(self):
            self.bytes = ''

        def write(self, bytes):
            self.bytes += bytes
    f = filelike()
    ujson.dump([1, 2, 3], f)
    self.assertEqual('[1,2,3]', f.bytes)