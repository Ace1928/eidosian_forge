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
def test_configure_add_invalid_type(self):
    self.config(cinder_volume_type='some_random_type')
    self._test_configure_add_invalid_type()