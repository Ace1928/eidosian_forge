import re
from keystoneauth1 import exceptions as ks_exceptions
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine.clients.os.keystone import heat_keystoneclient as hkc
def get_group_id(self, group, domain=None):
    if group is None:
        return None
    if not domain:
        group, domain = self.parse_entity_with_domain(group, 'KeystoneGroup')
    try:
        group_obj = self.client().client.groups.get(group)
        return group_obj.id
    except ks_exceptions.NotFound:
        group_list = self.client().client.groups.list(name=group, domain=domain)
        for group_obj in group_list:
            if group_obj.name == group:
                return group_obj.id
    raise exception.EntityNotFound(entity='KeystoneGroup', name=group)