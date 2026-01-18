from unittest import mock
from osprofiler.drivers import base
from osprofiler.tests import test
@mock.patch('oslo_utils.importutils.try_import')
def test_init_no_oslo_messaging(self, try_import_mock):
    try_import_mock.return_value = None
    self.assertRaises(ValueError, base.get_driver, 'messaging://', project='project', service='service', host='host', context={})