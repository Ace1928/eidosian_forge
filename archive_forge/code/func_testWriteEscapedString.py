import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def testWriteEscapedString(self):
    self.assertEqual('"\\u003cimg src=\'\\u0026amp;\'\\/\\u003e"', ujson.dumps("<img src='&amp;'/>", encode_html_chars=True))