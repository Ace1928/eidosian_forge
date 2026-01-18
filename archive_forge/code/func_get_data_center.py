from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def get_data_center(self):
    dc_name = self._module.params.get('data_center', None)
    if dc_name:
        system_service = self._connection.system_service()
        data_centers_service = system_service.data_centers_service()
        return data_centers_service.list(search='name=%s' % dc_name)[0]
    return dc_name