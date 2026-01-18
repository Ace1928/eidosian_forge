import abc
from keystoneauth1 import identity
from keystoneauth1 import session
from oslo_config import cfg
from oslo_context import context
from oslo_utils import uuidutils
from oslotest import base
from testtools import testcase
from castellan.common.credentials import keystone_password
from castellan.common.credentials import keystone_token
from castellan.common import exception
from castellan.key_manager import barbican_key_manager
from castellan.tests.functional import config
from castellan.tests.functional.key_manager import test_key_manager
from castellan.tests import utils
def test_consumer_add_secret_delete_force_parameter_true(self):
    """Consumer addition, secret deletion, 'force' parameter equals True

        Creates a secret, adds a consumer to it and deletes the secret,
        specifying the 'force' parameter as True.
        """
    key = test_key_manager._get_test_passphrase()
    self.assertIsNotNone(key)
    stored_id = self.key_mgr.store(self.ctxt, key)
    self.assertIsNotNone(stored_id)
    consumer = {'service': 'service1', 'resource_type': 'type1', 'resource_id': 'id1'}
    self.key_mgr.add_consumer(self.ctxt, stored_id, consumer)
    self.key_mgr.delete(self.ctxt, stored_id, True)