import binascii
from unittest import mock
from oslo_config import cfg
from castellan.common import exception
from castellan.common.objects import symmetric_key as key
from castellan import key_manager
from castellan.key_manager import not_implemented_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def test_get_fixed_key_fail_bad_context(self):
    self.assertRaises(exception.Forbidden, self.key_mgr.get, context=None, managed_object_id=self.fixed_key_id)