from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def get_export_fields(export):
    """ Return export fields dict """
    fields = export.get_fields()
    export_id = fields.get('id', None)
    permissions = fields.get('permissions', None)
    enabled = fields.get('enabled', None)
    field_dict = dict(id=export_id, permissions=permissions, enabled=enabled)
    return field_dict