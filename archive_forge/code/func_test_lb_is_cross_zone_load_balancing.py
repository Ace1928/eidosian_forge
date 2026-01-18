from tests.unit import unittest
from tests.compat import mock
from boto.ec2.elb import ELBConnection
from boto.ec2.elb import LoadBalancer
from boto.ec2.elb.attributes import LbAttributes
def test_lb_is_cross_zone_load_balancing(self):
    """Tests checking is_cross_zone_load_balancing."""
    mock_response, _, lb = self._setup_mock()
    tests = [(lb.is_cross_zone_load_balancing, [], True, ATTRIBUTE_GET_TRUE_CZL_RESPONSE), (lb.is_cross_zone_load_balancing, [], True, ATTRIBUTE_GET_FALSE_CZL_RESPONSE), (lb.is_cross_zone_load_balancing, [True], False, ATTRIBUTE_GET_FALSE_CZL_RESPONSE)]
    for method, args, result, response in tests:
        mock_response.read.return_value = response
        self.assertEqual(method(*args), result)