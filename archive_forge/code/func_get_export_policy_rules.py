from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import to_native
from ansible_collections.netapp.azure.plugins.module_utils.azure_rm_netapp_common import AzureRMNetAppModuleBase
from ansible_collections.netapp.azure.plugins.module_utils.netapp_module import NetAppModule
def get_export_policy_rules(self):
    ptypes = self.parameters.get('protocol_types')
    if ptypes is None:
        return None
    ptypes = [x.lower() for x in ptypes]
    if 'nfsv4.1' in ptypes:
        ptypes.append('nfsv41')
    if 'nfsv41' not in ptypes:
        return None
    options = dict(rule_index=1, allowed_clients='0.0.0.0/0', unix_read_write=True)
    if self.has_feature('ignore_change_ownership_mode') and self.sdk_version >= '4.0.0':
        options['chown_mode'] = None
    for protocol in ('cifs', 'nfsv3', 'nfsv41'):
        options[protocol] = protocol in ptypes
    return VolumePropertiesExportPolicy(rules=[ExportPolicyRule(**options)])