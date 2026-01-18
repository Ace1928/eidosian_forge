from tests.integration.route53 import Route53TestCase
from boto.compat import six
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets
def test_create_and_list_health_check(self):
    hc = HealthCheck(ip_addr='54.217.7.118', port=80, hc_type='HTTP', resource_path='/testing')
    result1 = self.conn.create_health_check(hc)
    hc = HealthCheck(ip_addr='54.217.7.119', port=80, hc_type='HTTP', resource_path='/testing')
    result2 = self.conn.create_health_check(hc)
    result = self.conn.get_list_health_checks()
    self.assertTrue(len(result['ListHealthChecksResponse']['HealthChecks']) > 1)
    self.conn.delete_health_check(result1['CreateHealthCheckResponse']['HealthCheck']['Id'])
    self.conn.delete_health_check(result2['CreateHealthCheckResponse']['HealthCheck']['Id'])