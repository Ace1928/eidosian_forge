from tests.unit import unittest
from tests.compat import mock
from boto.ec2.elb import ELBConnection
from boto.ec2.elb import LoadBalancer
from boto.ec2.elb.attributes import LbAttributes
def test_get_all_lb_attributes(self):
    """Tests getting the LbAttributes from the elb.connection."""
    mock_response, elb, _ = self._setup_mock()
    for response, attr_tests in ATTRIBUTE_TESTS:
        mock_response.read.return_value = response
        attributes = elb.get_all_lb_attributes('test_elb')
        self.assertTrue(isinstance(attributes, LbAttributes))
        self._verify_attributes(attributes, attr_tests)