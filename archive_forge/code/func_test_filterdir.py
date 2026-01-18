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
def test_filterdir(self):
    self.assertEqual(list(self.fs.filterdir('/', files=['*.py'])), [])
    self.fs.makedir('bar')
    self.fs.create('foo.txt')
    self.fs.create('foo.py')
    self.fs.create('foo.pyc')
    page1 = list(self.fs.filterdir('/', page=(None, 2)))
    page2 = list(self.fs.filterdir('/', page=(2, 4)))
    page3 = list(self.fs.filterdir('/', page=(4, 6)))
    self.assertEqual(len(page1), 2)
    self.assertEqual(len(page2), 2)
    self.assertEqual(len(page3), 0)
    names = [info.name for info in itertools.chain(page1, page2, page3)]
    self.assertEqual(set(names), {'foo.txt', 'foo.py', 'foo.pyc', 'bar'})
    dir_list = [info.name for info in self.fs.filterdir('/', files=['*.py'])]
    self.assertEqual(set(dir_list), {'bar', 'foo.py'})
    dir_list = [info.name for info in self.fs.filterdir('/', files=['*.py', '*.pyc'])]
    self.assertEqual(set(dir_list), {'bar', 'foo.py', 'foo.pyc'})
    dir_list = [info.name for info in self.fs.filterdir('/', exclude_dirs=['*'], files=['*.py', '*.pyc'])]
    self.assertEqual(set(dir_list), {'foo.py', 'foo.pyc'})
    dir_list = [info.name for info in self.fs.filterdir('/', exclude_files=['*'])]
    self.assertEqual(set(dir_list), {'bar'})
    with self.assertRaises(TypeError):
        dir_list = [info.name for info in self.fs.filterdir('/', files='*.py')]
    self.fs.makedir('baz')
    dir_list = [info.name for info in self.fs.filterdir('/', exclude_files=['*'], dirs=['??z'])]
    self.assertEqual(set(dir_list), {'baz'})
    with self.assertRaises(TypeError):
        dir_list = [info.name for info in self.fs.filterdir('/', exclude_files=['*'], dirs='*.py')]