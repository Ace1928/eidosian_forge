import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
@pytest.mark.parametrize('indent', list(range(65537, 65542)))
def test_dump_huge_indent(indent):
    ujson.encode({'a': True}, indent=indent)