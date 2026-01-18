from unittest import mock
from designateclient import exceptions as designate_exception
from heat.common import exception
from heat.engine.resources.openstack.designate import recordset
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_check_delete_complete(self):
    self.test_resource.resource_id = self._get_mock_resource()['id']
    self._mock_check_status_active()
    self.assertFalse(self.test_resource.check_delete_complete(self.test_resource.resource_id))
    self.assertTrue(self.test_resource.check_delete_complete(self.test_resource.resource_id))
    ex = self.assertRaises(exception.ResourceInError, self.test_resource.check_delete_complete, self.test_resource.resource_id)
    self.assertIn('Error in RecordSet', ex.message)