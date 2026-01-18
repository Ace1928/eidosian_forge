from tests.integration.route53 import Route53TestCase
from boto.compat import six
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets
def test_create_https_health_check_fqdn(self):
    hc = HealthCheck(ip_addr=None, port=443, hc_type='HTTPS', resource_path='/', fqdn='google.com')
    result = self.conn.create_health_check(hc)
    self.assertEquals(result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig'][u'Type'], 'HTTPS')
    self.assertEquals(result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig'][u'FullyQualifiedDomainName'], 'google.com')
    self.assertEquals(result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig'][u'Port'], '443')
    self.assertEquals(result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig'][u'ResourcePath'], '/')
    self.assertFalse('IPAddress' in result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig'])
    self.conn.delete_health_check(result['CreateHealthCheckResponse']['HealthCheck']['Id'])