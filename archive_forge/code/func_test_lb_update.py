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
def test_lb_update(self):
    self.conn.load_balancer.update_load_balancer(self.LB_ID, name=self.UPDATE_NAME)
    self.conn.load_balancer.wait_for_load_balancer(self.LB_ID, wait=self._wait_for_timeout)
    test_lb = self.conn.load_balancer.get_load_balancer(self.LB_ID)
    self.assertEqual(self.UPDATE_NAME, test_lb.name)
    self.conn.load_balancer.update_load_balancer(self.LB_ID, name=self.LB_NAME)
    self.conn.load_balancer.wait_for_load_balancer(self.LB_ID, wait=self._wait_for_timeout)
    test_lb = self.conn.load_balancer.get_load_balancer(self.LB_ID)
    self.assertEqual(self.LB_NAME, test_lb.name)