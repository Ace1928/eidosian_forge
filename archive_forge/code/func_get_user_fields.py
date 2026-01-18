from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def get_user_fields(user):
    """ Get user's fields """
    pools = user.get_owned_pools()
    pool_names = [pool.get_field('name') for pool in pools]
    fields = user.get_fields(from_cache=True, raw_value=True)
    field_dict = {'dn': fields.get('dn', None), 'email': fields.get('email', None), 'enabled': fields.get('enabled', None), 'id': user.id, 'ldap_id': fields.get('ldap_id', None), 'pools': pool_names, 'role': fields.get('role', None), 'roles': fields.get('roles', []), 'type': fields.get('type', None)}
    return field_dict