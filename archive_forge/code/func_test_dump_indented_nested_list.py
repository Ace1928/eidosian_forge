import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_dump_indented_nested_list():
    a = _a = []
    for i in range(20):
        _a.append(list(range(i)))
        _a = _a[-1]
        ujson.dumps(a, indent=i)