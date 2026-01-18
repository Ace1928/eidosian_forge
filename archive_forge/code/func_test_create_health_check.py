from tests.integration.route53 import Route53TestCase
from boto.compat import six
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets
def test_create_health_check(self):
    hc = HealthCheck(ip_addr='54.217.7.118', port=80, hc_type='HTTP', resource_path='/testing')
    result = self.conn.create_health_check(hc)
    self.assertEquals(result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig'][u'Type'], 'HTTP')
    self.assertEquals(result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig'][u'IPAddress'], '54.217.7.118')
    self.assertEquals(result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig'][u'Port'], '80')
    self.assertEquals(result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig'][u'ResourcePath'], '/testing')
    self.conn.delete_health_check(result['CreateHealthCheckResponse']['HealthCheck']['Id'])