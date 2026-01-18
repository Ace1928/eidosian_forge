import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def test_missing_tests(self):
    for loader in ImageLoader.loaders:
        key = 'test_{}'.format(loader.__name__)
        msg = 'Missing ImageLoader test case: {}'.format(key)
        self.assertTrue(hasattr(self, key), msg)
        self.assertTrue(callable(getattr(self, key)), msg)