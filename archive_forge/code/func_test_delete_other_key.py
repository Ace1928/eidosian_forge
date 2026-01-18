import binascii
from unittest import mock
from oslo_config import cfg
from castellan.common import exception
from castellan.common.objects import symmetric_key as key
from castellan import key_manager
from castellan.key_manager import not_implemented_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def test_delete_other_key(self):
    self.assertRaises(exception.KeyManagerError, self.key_mgr.delete, context=self.ctxt, managed_object_id=self.other_key_id)