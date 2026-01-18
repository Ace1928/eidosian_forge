import errno
import io
from unittest import mock
import sys
import uuid
import fixtures
from oslo_config import cfg
from oslo_utils import units
import glance_store as store
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit.cinder import test_cinder_base
from glance_store.tests.unit import test_store_capabilities as test_cap
from glance_store._drivers.cinder import store as cinder # noqa
def test_configure_add_authorization_failed(self):

    def fake_volume_type_check(name):
        raise cinder.exceptions.AuthorizationFailure(code=401)
    self.config(cinder_volume_type='some_type', group=self.store.backend_group)
    with mock.patch.object(self.store, 'get_cinderclient') as mocked_cc:
        mocked_cc.return_value = mock.MagicMock(volume_types=mock.MagicMock(find=fake_volume_type_check))
        self.assertRaises(cinder.exceptions.AuthorizationFailure, self.store.configure_add)