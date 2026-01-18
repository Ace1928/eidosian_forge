from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_adapters_uuids(self):
    missing_adapters = []
    adapters = [self.parameters['adapter_name']] + self.parameters.get('pair_adapters', [])
    for adapter in adapters:
        adapter_uuid = self.get_adapter_uuid(adapter)
        if adapter_uuid is None:
            missing_adapters.append(adapter)
        else:
            self.adapters_uuids[adapter] = adapter_uuid
    if missing_adapters:
        self.module.fail_json(msg='Error: Adapter(s) %s not exist' % ', '.join(missing_adapters))