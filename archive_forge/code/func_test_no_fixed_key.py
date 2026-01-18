import binascii
from unittest import mock
from oslo_config import cfg
from castellan.common import exception
from castellan.common.objects import symmetric_key as key
from castellan import key_manager
from castellan.key_manager import not_implemented_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def test_no_fixed_key(self):
    conf = self.conf
    conf.set_override('fixed_key', None, group='key_manager')
    key_mgr = key_manager.API(conf)
    self.assertNotEqual('MigrationKeyManager', type(key_mgr).__name__)
    self.assertRaises(exception.KeyManagerError, key_mgr.get, context=self.ctxt, managed_object_id=self.fixed_key_id)