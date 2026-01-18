import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_loadFileArgsError(self):
    self.assertRaises(TypeError, ujson.load, '[]')