import tempfile
import unittest
from boto.compat import StringIO, six, json
from textwrap import dedent
from boto.cloudfront.distribution import Distribution
def test_custom_policy_ip_address(self):
    """
        Test that a custom policy can be created with an IP address and
        an arbitrary URL.
        """
    url = 'http://1234567.cloudfront.com/*'
    ip_range = '192.168.0.1'
    policy = self.dist._custom_policy(url, ip_address=ip_range)
    policy = json.loads(policy)
    self.assertEqual(1, len(policy.keys()))
    statements = policy['Statement']
    self.assertEqual(1, len(statements))
    statement = statements[0]
    resource = statement['Resource']
    self.assertEqual(url, resource)
    condition = statement['Condition']
    self.assertEqual(2, len(condition.keys()))
    ip_address = condition['IpAddress']
    self.assertTrue('DateLessThan' in condition)
    self.assertEqual(1, len(ip_address.keys()))
    source_ip = ip_address['AWS:SourceIp']
    self.assertEqual('%s/32' % ip_range, source_ip)