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
def setup_single_store(self):
    glance_store.register_opts(CONF)
    self.config(show_multiple_locations=True)
    self.config(show_image_direct_url=True)
    self.config(default_store='cinder', group='glance_store')
    self.config(stores=['http', 'swift', 'cinder'], group='glance_store')
    self.config(cinder_volume_type='fast', group='glance_store')
    self.config(cinder_store_user_name='fake_user', group='glance_store')
    self.config(cinder_store_password='fake_pass', group='glance_store')
    self.config(cinder_store_project_name='fake_project', group='glance_store')
    self.config(cinder_store_auth_address='http://auth_addr', group='glance_store')
    glance_store.create_stores(CONF)