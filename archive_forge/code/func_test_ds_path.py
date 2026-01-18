from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_ds_path(self):
    p = datastore.DatastorePath('dsname', 'a/b/c', 'file.iso')
    self.assertEqual('[dsname] a/b/c/file.iso', str(p))
    self.assertEqual('a/b/c/file.iso', p.rel_path)
    self.assertEqual('a/b/c', p.parent.rel_path)
    self.assertEqual('[dsname] a/b/c', str(p.parent))
    self.assertEqual('dsname', p.datastore)
    self.assertEqual('file.iso', p.basename)
    self.assertEqual('a/b/c', p.dirname)