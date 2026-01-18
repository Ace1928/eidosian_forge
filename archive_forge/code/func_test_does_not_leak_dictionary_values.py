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
def test_does_not_leak_dictionary_values(self):
    import gc
    gc.collect()
    value = ['abc']
    data = {'1': value}
    ref_count = sys.getrefcount(value)
    ujson.dumps(data)
    self.assertEqual(ref_count, sys.getrefcount(value))