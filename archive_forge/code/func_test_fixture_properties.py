from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.tests import utils as test_utils
def test_fixture_properties(self):
    conf = self.messaging_conf.conf
    self.messaging_conf.transport_url = 'fake:/vhost'
    self.assertEqual('fake:/vhost', self.messaging_conf.transport_url)
    self.assertEqual('fake:/vhost', conf.transport_url)