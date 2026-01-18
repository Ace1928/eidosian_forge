import errno
import io
from unittest import mock
import sys
import uuid
from oslo_utils import units
from glance_store import exceptions
from glance_store.tests import base
from glance_store.tests.unit.cinder import test_cinder_base
from glance_store.tests.unit import test_store_capabilities
from glance_store._drivers.cinder import store as cinder # noqa
def test_cinder_configure_add(self):
    self.assertRaises(exceptions.BadStoreConfiguration, self.store._check_context, None)
    self.assertRaises(exceptions.BadStoreConfiguration, self.store._check_context, mock.MagicMock(service_catalog=None))
    self.store._check_context(mock.MagicMock(service_catalog='fake'))