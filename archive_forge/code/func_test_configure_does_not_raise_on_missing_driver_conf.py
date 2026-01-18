from unittest import mock
from oslo_config import cfg
import glance_store as store
from glance_store import backend
from glance_store import location
from glance_store import multi_backend
from glance_store.tests import base
@mock.patch.object(store.driver, 'LOG')
def test_configure_does_not_raise_on_missing_driver_conf(self, mock_log):
    self.config(stores=['file'], group='glance_store')
    self.config(filesystem_store_datadir=None, group='glance_store')
    self.config(filesystem_store_datadirs=None, group='glance_store')
    for __, store_instance in backend._load_stores(self.conf):
        store_instance.configure()
        mock_log.warning.assert_called_once_with("Failed to configure store correctly: Store filesystem could not be configured correctly. Reason: Specify at least 'filesystem_store_datadir' or 'filesystem_store_datadirs' option Disabling add method.")