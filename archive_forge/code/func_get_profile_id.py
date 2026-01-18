from openstack import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients.os import openstacksdk as sdk_plugin
from heat.engine import constraints
def get_profile_id(self, profile_name):
    profile = self.client().get_profile(profile_name)
    return profile.id