from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_ds_path_parse(self):
    p = datastore.DatastorePath.parse('[dsname]')
    self.assertEqual('dsname', p.datastore)
    self.assertEqual('', p.rel_path)
    p = datastore.DatastorePath.parse('[dsname] folder')
    self.assertEqual('dsname', p.datastore)
    self.assertEqual('folder', p.rel_path)
    p = datastore.DatastorePath.parse('[dsname] folder/file')
    self.assertEqual('dsname', p.datastore)
    self.assertEqual('folder/file', p.rel_path)
    for p in [None, '']:
        self.assertRaises(ValueError, datastore.DatastorePath.parse, p)
    for p in ['bad path', '/a/b/c', 'a/b/c']:
        self.assertRaises(IndexError, datastore.DatastorePath.parse, p)