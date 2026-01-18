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
def test_health_monitor_list(self):
    names = [hm.name for hm in self.conn.load_balancer.health_monitors()]
    self.assertIn(self.HM_NAME, names)