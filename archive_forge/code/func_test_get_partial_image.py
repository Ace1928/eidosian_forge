import io
from unittest import mock
from oslo_config import cfg
from oslo_utils import units
import glance_store as store
from glance_store._drivers import rbd as rbd_store
from glance_store import exceptions
from glance_store import location as g_location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_get_partial_image(self):
    loc = g_location.Location('test_rbd_store', rbd_store.StoreLocation, self.conf, store_specs=self.store_specs)
    self.assertRaises(exceptions.StoreRandomGetNotSupported, self.store.get, loc, chunk_size=1)