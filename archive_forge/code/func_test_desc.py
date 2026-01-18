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
def test_desc(self):
    self.fs.create('foo')
    description = self.fs.desc('foo')
    self.assertIsInstance(description, text_type)
    self.fs.makedir('dir')
    self.fs.desc('dir')
    self.fs.desc('/')
    self.fs.desc('')
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.desc('bar')