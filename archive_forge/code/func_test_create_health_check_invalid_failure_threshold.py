from tests.integration.route53 import Route53TestCase
from boto.compat import six
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets
def test_create_health_check_invalid_failure_threshold(self):
    """
        Test that health checks cannot be created with an invalid
        'failure_threshold'.
        """
    self.assertRaises(AttributeError, lambda: HealthCheck(**self.health_check_params(failure_threshold=0)))
    self.assertRaises(AttributeError, lambda: HealthCheck(**self.health_check_params(failure_threshold=11)))