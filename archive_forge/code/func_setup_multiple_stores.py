import socket
from unittest import mock
import uuid
from cinderclient.v3 import client as cinderclient
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import strutils
from glance.common import wsgi
from glance.tests import functional
@mock.patch.object(cinderclient, 'Client')
def setup_multiple_stores(self, mock_client):
    """Configures multiple backend stores.

        This configures the API with two cinder stores (store1 and
        store2) as well as a os_glance_staging_store for
        imports.

        """
    self.config(show_multiple_locations=True)
    self.config(show_image_direct_url=True)
    self.config(enabled_backends={'store1': 'cinder', 'store2': 'cinder'})
    glance_store.register_store_opts(CONF, reserved_stores=wsgi.RESERVED_STORES)
    self.config(default_backend='store1', group='glance_store')
    self.config(cinder_volume_type='fast', group='store1')
    self.config(cinder_store_user_name='fake_user', group='store1')
    self.config(cinder_store_password='fake_pass', group='store1')
    self.config(cinder_store_project_name='fake_project', group='store1')
    self.config(cinder_store_auth_address='http://auth_addr', group='store1')
    self.config(cinder_volume_type='reliable', group='store2')
    self.config(cinder_store_user_name='fake_user', group='store2')
    self.config(cinder_store_password='fake_pass', group='store2')
    self.config(cinder_store_project_name='fake_project', group='store2')
    self.config(cinder_store_auth_address='http://auth_addr', group='store2')
    self.config(filesystem_store_datadir=self._store_dir('staging'), group='os_glance_staging_store')
    glance_store.create_multi_stores(CONF, reserved_stores=wsgi.RESERVED_STORES)
    glance_store.verify_store()