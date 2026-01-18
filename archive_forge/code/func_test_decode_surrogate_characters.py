import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
@pytest.mark.parametrize('test_input, expected', [('"\\uD83D\\uDCA9"', '💩'), ('"a\\uD83D\\uDCA9b"', 'a💩b'), ('"\\uD800"', '\ud800'), ('"a\\uD800b"', 'a\ud800b'), ('"\\uDEAD"', '\udead'), ('"a\\uDEADb"', 'a\udeadb'), ('"\\uD83D\\uD83D\\uDCA9"', '\ud83d💩'), ('"\\uDCA9\\uD83D\\uDCA9"', '\udca9💩'), ('"\\uD83D\\uDCA9\\uD83D"', '💩\ud83d'), ('"\\uD83D\\uDCA9\\uDCA9"', '💩\udca9'), ('"\\uD83D \\uDCA9"', '\ud83d \udca9'), ('"\ud800"', '\ud800'), ('"\udead"', '\udead'), ('"\ud800a\udead"', '\ud800a\udead'), ('"\ud83d\udca9"', '\ud83d\udca9')])
def test_decode_surrogate_characters(test_input, expected):
    assert ujson.loads(test_input) == expected
    assert ujson.loads(test_input.encode('utf-8', 'surrogatepass')) == expected
    assert json.loads(test_input) == expected