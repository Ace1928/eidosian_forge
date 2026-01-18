from tests.unit import unittest
from tests.compat import mock
from boto.ec2.elb import ELBConnection
from boto.ec2.elb import LoadBalancer
from boto.ec2.elb.attributes import LbAttributes
def test_lb_enable_cross_zone_load_balancing(self):
    """Tests enabling cross zone balancing from LoadBalancer."""
    mock_response, elb, lb = self._setup_mock()
    mock_response.read.return_value = ATTRIBUTE_SET_RESPONSE
    self.assertTrue(lb.enable_cross_zone_load_balancing())
    elb.make_request.assert_called_with(*ATTRIBUTE_SET_CZL_TRUE_REQUEST)