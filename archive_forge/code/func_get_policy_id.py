from openstack import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients.os import openstacksdk as sdk_plugin
from heat.engine import constraints
def get_policy_id(self, policy_name):
    policy = self.client().get_policy(policy_name)
    return policy.id