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
def test_makedirs(self):
    self.assertFalse(self.fs.exists('foo'))
    self.fs.makedirs('foo')
    self.assertEqual(self.fs.gettype('foo'), ResourceType.directory)
    self.fs.makedirs('foo/bar/baz')
    self.assertTrue(self.fs.isdir('foo/bar'))
    self.assertTrue(self.fs.isdir('foo/bar/baz'))
    with self.assertRaises(errors.DirectoryExists):
        self.fs.makedirs('foo/bar/baz')
    self.fs.makedirs('foo/bar/baz', recreate=True)
    self.fs.writebytes('foo.bin', b'test')
    with self.assertRaises(errors.DirectoryExpected):
        self.fs.makedirs('foo.bin/bar')
    with self.assertRaises(errors.DirectoryExpected):
        self.fs.makedirs('foo.bin/bar/baz/egg')