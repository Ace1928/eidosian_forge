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
def test_getsyspath(self):
    self.fs.create('foo')
    try:
        syspath = self.fs.getsyspath('foo')
    except errors.NoSysPath:
        self.assertFalse(self.fs.hassyspath('foo'))
    else:
        self.assertIsInstance(syspath, text_type)
        self.assertIsInstance(self.fs.getospath('foo'), bytes)
        self.assertTrue(self.fs.hassyspath('foo'))
    self.fs.hassyspath('a/b/c/foo/bar')