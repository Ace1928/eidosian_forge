from tests.unit import unittest
from tests.compat import mock
from boto.ec2.elb import ELBConnection
from boto.ec2.elb import LoadBalancer
from boto.ec2.elb.attributes import LbAttributes
def test_modify_lb_attribute(self):
    """Tests setting the attributes from elb.connection."""
    mock_response, elb, _ = self._setup_mock()
    tests = [('crossZoneLoadBalancing', True, ATTRIBUTE_SET_CZL_TRUE_REQUEST), ('crossZoneLoadBalancing', False, ATTRIBUTE_SET_CZL_FALSE_REQUEST)]
    for attr, value, args in tests:
        mock_response.read.return_value = ATTRIBUTE_SET_RESPONSE
        result = elb.modify_lb_attribute('test_elb', attr, value)
        self.assertTrue(result)
        elb.make_request.assert_called_with(*args)