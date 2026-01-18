from __future__ import absolute_import, unicode_literals
import io
import itertools
import json
import os
import six
import time
import unittest
import warnings
from datetime import datetime
from six import text_type
import fs.copy
import fs.move
from fs import ResourceType, Seek, errors, glob, walk
from fs.opener import open_fs
from fs.subfs import ClosingSubFS, SubFS
def test_invalid_chars(self):
    with self.assertRaises(errors.InvalidCharsInPath):
        self.fs.open('invalid\x00file', 'wb')
    with self.assertRaises(errors.InvalidCharsInPath):
        self.fs.validatepath('invalid\x00file')