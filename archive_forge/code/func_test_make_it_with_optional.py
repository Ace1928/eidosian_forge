from openstack.network.v2 import network_ip_availability
from openstack.tests.unit import base
def test_make_it_with_optional(self):
    sot = network_ip_availability.NetworkIPAvailability(**EXAMPLE_WITH_OPTIONAL)
    self.assertEqual(EXAMPLE_WITH_OPTIONAL['network_id'], sot.network_id)
    self.assertEqual(EXAMPLE_WITH_OPTIONAL['network_name'], sot.network_name)
    self.assertEqual(EXAMPLE_WITH_OPTIONAL['subnet_ip_availability'], sot.subnet_ip_availability)
    self.assertEqual(EXAMPLE_WITH_OPTIONAL['project_id'], sot.project_id)
    self.assertEqual(EXAMPLE_WITH_OPTIONAL['total_ips'], sot.total_ips)
    self.assertEqual(EXAMPLE_WITH_OPTIONAL['used_ips'], sot.used_ips)