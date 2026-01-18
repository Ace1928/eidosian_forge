import hashlib
import io
from unittest import mock
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import rbd as rbd_store
from glance_store import exceptions
from glance_store import location as g_location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils as test_utils
@mock.patch('oslo_utils.eventletutils.is_monkey_patched')
def test_create_image_in_native_thread(self, mock_patched):
    mock_patched.return_value = True
    fsid = 'fake'
    features = '3'
    conf_get_mock = mock.Mock(return_value=features)
    conn = mock.Mock(conf_get=conf_get_mock)
    ioctxt = mock.sentinel.ioctxt
    name = '1'
    size = 1024
    order = 3
    fake_proxy = mock.MagicMock()
    fake_rbd = mock.MagicMock()
    with mock.patch.object(rbd_store.tpool, 'Proxy') as tpool_mock, mock.patch.object(rbd_store.rbd, 'RBD') as rbd_mock:
        tpool_mock.return_value = fake_proxy
        rbd_mock.return_value = fake_rbd
        location = self.store._create_image(fsid, conn, ioctxt, name, size, order)
        self.assertEqual(fsid, location.specs['fsid'])
        self.assertEqual(rbd_store.DEFAULT_POOL, location.specs['pool'])
        self.assertEqual(name, location.specs['image'])
        self.assertEqual(rbd_store.DEFAULT_SNAPNAME, location.specs['snapshot'])
    tpool_mock.assert_called_once_with(fake_rbd)
    fake_proxy.create.assert_called_once_with(ioctxt, name, size, order, old_format=False, features=3)