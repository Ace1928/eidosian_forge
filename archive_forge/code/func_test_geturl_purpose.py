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
def test_geturl_purpose(self):
    """Check an unknown purpose raises a NoURL error."""
    self.fs.create('foo')
    with self.assertRaises(errors.NoURL):
        self.fs.geturl('foo', purpose='__nosuchpurpose__')