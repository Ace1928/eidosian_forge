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
def test_getsize(self):
    self.fs.writebytes('empty', b'')
    self.fs.writebytes('one', b'a')
    self.fs.writebytes('onethousand', ('b' * 1000).encode('ascii'))
    self.assertEqual(self.fs.getsize('empty'), 0)
    self.assertEqual(self.fs.getsize('one'), 1)
    self.assertEqual(self.fs.getsize('onethousand'), 1000)
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.getsize('doesnotexist')