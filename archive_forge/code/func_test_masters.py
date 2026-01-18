from unittest import mock
from heat.common import exception
from heat.engine.resources.openstack.designate import zone
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_masters(self):
    self.do_test({'heat_template_version': '2015-04-30', 'resources': {'test_resource': {'type': 'OS::Designate::Zone', 'properties': {'name': 'test-zone.com', 'description': 'Test zone', 'ttl': 3600, 'email': 'abc@test-zone.com', 'type': 'SECONDARY', 'masters': self.primaries}}}})