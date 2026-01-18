from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
import traceback
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def get_export_client_fields(export, client_name):
    """ Get export client fields """
    fields = export.get_fields()
    permissions = fields.get('permissions', None)
    for munched_perm in permissions:
        perm = unmunchify(munched_perm)
        if perm['client'] == client_name:
            field_dict = dict(access_mode=perm['access'], no_root_squash=perm['no_root_squash'])
            return field_dict
    raise AssertionError(f'No client {client_name} match to exports found')