from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_ds_invalid(self):
    self.assertRaises(ValueError, datastore.Datastore, None, 'ds_name')
    self.assertRaises(ValueError, datastore.Datastore, 'fake_ref', None)