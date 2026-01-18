import os
import sys
from unittest import mock
import ddt
from glance_store import exceptions
from glance_store.tests.unit.cinder import test_base as test_base_connector
from glance_store._drivers.cinder import store as cinder # noqa
from glance_store._drivers.cinder import nfs # noqa
def test_disconnect_volume(self):
    fake_hash = 'fake_hash'
    fake_path = {'path': os.path.join(self.mountpath, fake_hash, self.connection_info['name'])}
    mount_path, vol_name = fake_path['path'].rsplit('/', 1)
    self.connector.disconnect_volume(fake_path)
    nfs.mount.umount.assert_called_once_with(vol_name, mount_path, self.connector.host, self.connector.root_helper)