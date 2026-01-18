from unittest import mock
import uuid
from openstack.load_balancer.v2 import _proxy
from openstack.load_balancer.v2 import amphora
from openstack.load_balancer.v2 import availability_zone
from openstack.load_balancer.v2 import availability_zone_profile
from openstack.load_balancer.v2 import flavor
from openstack.load_balancer.v2 import flavor_profile
from openstack.load_balancer.v2 import health_monitor
from openstack.load_balancer.v2 import l7_policy
from openstack.load_balancer.v2 import l7_rule
from openstack.load_balancer.v2 import listener
from openstack.load_balancer.v2 import load_balancer as lb
from openstack.load_balancer.v2 import member
from openstack.load_balancer.v2 import pool
from openstack.load_balancer.v2 import provider
from openstack.load_balancer.v2 import quota
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
def test_l7_rule_delete(self):
    self.verify_delete(self.proxy.delete_l7_rule, l7_rule.L7Rule, ignore_missing=True, method_kwargs={'l7_policy': self.L7_POLICY_ID}, expected_kwargs={'l7policy_id': self.L7_POLICY_ID})