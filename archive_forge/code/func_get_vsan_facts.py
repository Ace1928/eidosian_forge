from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.common.text.formatters import bytes_to_human
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def get_vsan_facts(self):
    config_mgr = self.host.configManager.vsanSystem
    ret = {'vsan_cluster_uuid': None, 'vsan_node_uuid': None, 'vsan_health': 'unknown'}
    if config_mgr is None:
        return ret
    try:
        status = config_mgr.QueryHostStatus()
    except (vmodl.fault.HostNotConnected, vmodl.fault.HostNotReachable):
        return {'vsan_cluster_uuid': 'NA', 'vsan_node_uuid': 'NA', 'vsan_health': 'NA'}
    except Exception as err:
        self.module.fail_json(msg='Unable to query VSAN status due to %s' % to_native(err))
    return {'vsan_cluster_uuid': status.uuid, 'vsan_node_uuid': status.nodeUuid, 'vsan_health': status.health}