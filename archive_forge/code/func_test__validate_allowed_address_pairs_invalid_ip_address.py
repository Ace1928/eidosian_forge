from unittest import mock
from oslo_config import cfg
from webob import exc
from neutron_lib.api.validators import allowedaddresspairs as validator
from neutron_lib.exceptions import allowedaddresspairs as addr_exc
from neutron_lib.tests import _base as base
@mock.patch.object(cfg, 'CONF')
def test__validate_allowed_address_pairs_invalid_ip_address(self, mock_conf):
    mock_conf.max_allowed_address_pair = 3
    self.assertRaisesRegex(exc.HTTPBadRequest, 'is not a valid IP address', validator._validate_allowed_address_pairs, [{'ip_address': '192.168.1.a'}, {'ip_address': '192.168.1.2'}])