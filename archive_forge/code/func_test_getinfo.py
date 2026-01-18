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
def test_getinfo(self):
    root_info = self.fs.getinfo('/')
    self.assertEqual(root_info.name, '')
    self.assertTrue(root_info.is_dir)
    self.assertIn('basic', root_info.namespaces)
    self.fs.writebytes('foo', b'bar')
    self.fs.makedir('dir')
    info = self.fs.getinfo('foo').raw
    self.assertIn('basic', info)
    self.assertIsInstance(info['basic']['name'], text_type)
    self.assertEqual(info['basic']['name'], 'foo')
    self.assertFalse(info['basic']['is_dir'])
    info = self.fs.getinfo('dir').raw
    self.assertIn('basic', info)
    self.assertEqual(info['basic']['name'], 'dir')
    self.assertTrue(info['basic']['is_dir'])
    info = self.fs.getinfo('foo', namespaces=['details']).raw
    self.assertIn('basic', info)
    self.assertIsInstance(info, dict)
    self.assertEqual(info['details']['size'], 3)
    self.assertEqual(info['details']['type'], int(ResourceType.file))
    self.assertEqual(info, self.fs.getdetails('foo').raw)
    try:
        json.dumps(info)
    except (TypeError, ValueError):
        raise AssertionError('info should be JSON serializable')
    no_info = self.fs.getinfo('foo', '__nosuchnamespace__').raw
    self.assertIsInstance(no_info, dict)
    self.assertEqual(no_info['basic'], {'name': 'foo', 'is_dir': False})
    info = self.fs.getinfo('foo', namespaces=['access', 'stat', 'details'])
    if 'details' in info.namespaces:
        details = info.raw['details']
        self.assertIsInstance(details.get('accessed'), (type(None), int, float))
        self.assertIsInstance(details.get('modified'), (type(None), int, float))
        self.assertIsInstance(details.get('created'), (type(None), int, float))
        self.assertIsInstance(details.get('metadata_changed'), (type(None), int, float))