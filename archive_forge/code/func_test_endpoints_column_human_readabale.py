from unittest import mock
from openstackclient.identity.v3 import catalog
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
def test_endpoints_column_human_readabale(self):
    col = catalog.EndpointsColumn(self.fake_service['endpoints'])
    self.assertEqual('onlyone\n  public: https://public.example.com\nonlyone\n  admin: https://admin.example.com\n<none>\n  internal: https://internal.example.com\n<none>\n  none: https://none.example.com\n', col.human_readable())