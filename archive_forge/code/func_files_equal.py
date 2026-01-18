import contextlib
import os.path
import sys
import tempfile
import unittest
from io import open
from os.path import join as pjoin
from ..Dependencies import extended_iglob
def files_equal(self, pattern, expected_files):
    expected_files = sorted(expected_files)
    matched_files = sorted((path.replace('/', os.sep) for path in extended_iglob(pattern)))
    self.assertListEqual(matched_files, expected_files)
    if os.sep == '\\' and '/' in pattern:
        matched_files = sorted(extended_iglob(pattern.replace('/', '\\')))
        self.assertListEqual(matched_files, expected_files)