import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
@pytest.mark.parametrize('indent', [0, 1, 2, 4, 5, 8, 49])
def test_issue_334(indent):
    path = Path(__file__).with_name('334-reproducer.json')
    a = ujson.loads(path.read_bytes())
    ujson.dumps(a, indent=indent)