from unittest import mock
from saharaclient.osc import utils
from saharaclient.tests.unit import base
def test_get_resource_id(self):
    manager = mock.Mock()
    uuid = '82065b4d-2c79-420d-adc3-310de275e922'
    manager.find_unique.return_value = mock.Mock(id=uuid)
    res = utils.get_resource_id(manager, uuid)
    self.assertEqual(uuid, res)
    manager.get.assert_not_called()
    manager.find_unique.assert_not_called()
    res = utils.get_resource_id(manager, 'name')
    manager.find_unique.assert_called_once_with(name='name')
    self.assertEqual(uuid, res)