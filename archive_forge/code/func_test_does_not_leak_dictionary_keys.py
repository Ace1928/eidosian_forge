import decimal
import json
import math
import sys
import unittest
import pytest
from io import StringIO
from pathlib import Path
from srsly import ujson
@unittest.skipIf(not hasattr(sys, 'getrefcount') == True, reason='test requires sys.refcount')
def test_does_not_leak_dictionary_keys(self):
    import gc
    gc.collect()
    key1 = '1'
    key2 = '1'
    value1 = ['abc']
    value2 = [1, 2, 3]
    data = {key1: value1, key2: value2}
    ref_count1 = sys.getrefcount(key1)
    ref_count2 = sys.getrefcount(key2)
    ujson.dumps(data)
    self.assertEqual(ref_count1, sys.getrefcount(key1))
    self.assertEqual(ref_count2, sys.getrefcount(key2))