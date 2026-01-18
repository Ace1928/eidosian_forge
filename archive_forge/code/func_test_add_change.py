from tests.compat import unittest
from tests.integration.route53 import Route53TestCase
from boto.route53.record import ResourceRecordSets
def test_add_change(self):
    rrs = ResourceRecordSets(self.conn, self.zone.id)
    created = rrs.add_change('CREATE', 'vpn.%s.' % self.base_domain, 'A')
    created.add_value('192.168.0.25')
    rrs.commit()
    rrs = ResourceRecordSets(self.conn, self.zone.id)
    deleted = rrs.add_change('DELETE', 'vpn.%s.' % self.base_domain, 'A')
    deleted.add_value('192.168.0.25')
    rrs.commit()