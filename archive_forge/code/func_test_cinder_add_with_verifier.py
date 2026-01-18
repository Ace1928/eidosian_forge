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
def test_cinder_add_with_verifier(self):
    fake_volume = mock.MagicMock(id=str(uuid.uuid4()), status='available', size=1)
    volume_file = io.BytesIO()
    verifier = mock.MagicMock()
    self._test_cinder_add(fake_volume, volume_file, 1, verifier)
    verifier.update.assert_called_with(b'*' * units.Ki)