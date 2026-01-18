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
def test_bin_files(self):
    with self.fs.openbin('foo1', 'wb') as f:
        text_type(f)
        repr(f)
        f.write(b'a')
        f.write(b'b')
        f.write(b'c')
    self.assert_bytes('foo1', b'abc')
    with self.fs.openbin('foo2', 'wb') as f:
        f.writelines([b'hello\n', b'world'])
    self.assert_bytes('foo2', b'hello\nworld')
    with self.fs.openbin('foo2') as f:
        self.assertEqual(f.readline(), b'hello\n')
        self.assertEqual(f.readline(), b'world')
    with self.fs.openbin('foo2') as f:
        lines = f.readlines()
    self.assertEqual(lines, [b'hello\n', b'world'])
    with self.fs.openbin('foo2') as f:
        lines = list(f)
    self.assertEqual(lines, [b'hello\n', b'world'])
    with self.fs.openbin('foo2') as f:
        lines = []
        for line in f:
            lines.append(line)
    self.assertEqual(lines, [b'hello\n', b'world'])
    with self.fs.openbin('foo2') as f:
        print(repr(f))
        self.assertEqual(next(f), b'hello\n')
    with self.fs.open('foo2', 'r+b') as f:
        f.truncate(3)
    self.assertEqual(self.fs.getsize('foo2'), 3)
    self.assert_bytes('foo2', b'hel')