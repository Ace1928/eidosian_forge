from tests.unit import unittest
from tests.compat import mock
from boto.ec2.elb import ELBConnection
from boto.ec2.elb import LoadBalancer
from boto.ec2.elb.attributes import LbAttributes
def test_get_lb_attribute(self):
    """Tests getting a single attribute from elb.connection."""
    mock_response, elb, _ = self._setup_mock()
    tests = [('crossZoneLoadBalancing', True, ATTRIBUTE_GET_TRUE_CZL_RESPONSE), ('crossZoneLoadBalancing', False, ATTRIBUTE_GET_FALSE_CZL_RESPONSE)]
    for attr, value, response in tests:
        mock_response.read.return_value = response
        status = elb.get_lb_attribute('test_elb', attr)
        self.assertEqual(status, value)