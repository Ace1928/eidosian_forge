from tests.integration.route53 import Route53TestCase
from boto.compat import six
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets
def test_create_health_check_failure_threshold(self):
    hc_params = self.health_check_params(failure_threshold=1)
    hc = HealthCheck(**hc_params)
    result = self.conn.create_health_check(hc)
    hc_config = result[u'CreateHealthCheckResponse'][u'HealthCheck'][u'HealthCheckConfig']
    self.assertEquals(hc_config[u'FailureThreshold'], six.text_type(hc_params['failure_threshold']))
    self.conn.delete_health_check(result['CreateHealthCheckResponse']['HealthCheck']['Id'])