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
def assert_text(self, path, contents):
    """Assert a file contains the given text.

        Arguments:
            path (str): A path on the filesystem.
            contents (str): Text to compare.

        """
    assert isinstance(contents, text_type)
    with self.fs.open(path, 'rt') as f:
        data = f.read()
    self.assertEqual(data, contents)
    self.assertIsInstance(data, text_type)