from tests.integration.route53 import Route53TestCase
from boto.compat import six
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets
def test_create_resource_record_set(self):
    hc = HealthCheck(ip_addr='54.217.7.118', port=80, hc_type='HTTP', resource_path='/testing')
    result = self.conn.create_health_check(hc)
    records = ResourceRecordSets(connection=self.conn, hosted_zone_id=self.zone.id, comment='Create DNS entry for test')
    change = records.add_change('CREATE', 'unittest.%s.' % self.base_domain, 'A', ttl=30, identifier='test', weight=1, health_check=result['CreateHealthCheckResponse']['HealthCheck']['Id'])
    change.add_value('54.217.7.118')
    records.commit()
    records = ResourceRecordSets(self.conn, self.zone.id)
    deleted = records.add_change('DELETE', 'unittest.%s.' % self.base_domain, 'A', ttl=30, identifier='test', weight=1, health_check=result['CreateHealthCheckResponse']['HealthCheck']['Id'])
    deleted.add_value('54.217.7.118')
    records.commit()