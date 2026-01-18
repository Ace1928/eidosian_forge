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
def test_does_not_leak_dictionary_string_key(self):
    import gc
    gc.collect()
    key1 = '1'
    value1 = 1
    data = {key1: value1}
    ref_count1 = sys.getrefcount(key1)
    ujson.dumps(data)
    self.assertEqual(ref_count1, sys.getrefcount(key1))