import os
import sys
from unittest import mock
import ddt
from glance_store import exceptions
from glance_store.tests.unit.cinder import test_base as test_base_connector
from glance_store._drivers.cinder import store as cinder # noqa
from glance_store._drivers.cinder import nfs # noqa
def test_extend_volume(self):
    self.assertRaises(NotImplementedError, self.connector.extend_volume)