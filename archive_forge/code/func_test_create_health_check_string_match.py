from tests.integration.route53 import Route53TestCase
from boto.compat import six
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets
def test_create_health_check_string_match(self):
    hc = HealthCheck(ip_addr='54.217.7.118', port=80, hc_type='HTTP_STR_MATCH', resource_path='/testing', string_match='test')
    result = self.conn.create_health_check(hc)
    self.assertEquals(result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig'][u'Type'], 'HTTP_STR_MATCH')
    self.assertEquals(result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig'][u'IPAddress'], '54.217.7.118')
    self.assertEquals(result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig'][u'Port'], '80')
    self.assertEquals(result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig'][u'ResourcePath'], '/testing')
    self.assertEquals(result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig'][u'SearchString'], 'test')
    self.conn.delete_health_check(result['CreateHealthCheckResponse']['HealthCheck']['Id'])