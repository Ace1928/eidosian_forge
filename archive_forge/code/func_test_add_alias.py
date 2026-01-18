import time
from tests.compat import unittest
from boto.route53.connection import Route53Connection
from boto.route53.record import ResourceRecordSets
from boto.route53.exception import DNSServerError
def test_add_alias(self):
    base_record = dict(name='alias.%s.' % self.base_domain, type='A', alias_evaluate_target_health=False, alias_dns_name='target.%s' % self.base_domain, alias_hosted_zone_id=self.zone.id, identifier='boto:TestRoute53AliasResourceRecordSets')
    rrs = ResourceRecordSets(self.conn, self.zone.id)
    rrs.add_change(action='UPSERT', **base_record)
    rrs.commit()
    rrs = ResourceRecordSets(self.conn, self.zone.id)
    rrs.add_change(action='DELETE', **base_record)
    rrs.commit()