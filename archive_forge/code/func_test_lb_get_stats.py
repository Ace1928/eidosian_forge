from openstack.load_balancer.v2 import availability_zone
from openstack.load_balancer.v2 import availability_zone_profile
from openstack.load_balancer.v2 import flavor
from openstack.load_balancer.v2 import flavor_profile
from openstack.load_balancer.v2 import health_monitor
from openstack.load_balancer.v2 import l7_policy
from openstack.load_balancer.v2 import l7_rule
from openstack.load_balancer.v2 import listener
from openstack.load_balancer.v2 import load_balancer
from openstack.load_balancer.v2 import member
from openstack.load_balancer.v2 import pool
from openstack.load_balancer.v2 import quota
from openstack.tests.functional import base
def test_lb_get_stats(self):
    test_lb_stats = self.conn.load_balancer.get_load_balancer_statistics(self.LB_ID)
    self.assertEqual(0, test_lb_stats.active_connections)
    self.assertEqual(0, test_lb_stats.bytes_in)
    self.assertEqual(0, test_lb_stats.bytes_out)
    self.assertEqual(0, test_lb_stats.request_errors)
    self.assertEqual(0, test_lb_stats.total_connections)