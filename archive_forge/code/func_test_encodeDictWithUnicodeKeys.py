import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
def test_encodeDictWithUnicodeKeys(self):
    input = {'key1': 'value1', 'key1': 'value1', 'key1': 'value1', 'key1': 'value1', 'key1': 'value1', 'key1': 'value1'}
    ujson.encode(input)
    input = {'بن': 'value1', 'بن': 'value1', 'بن': 'value1', 'بن': 'value1', 'بن': 'value1', 'بن': 'value1', 'بن': 'value1'}
    ujson.encode(input)