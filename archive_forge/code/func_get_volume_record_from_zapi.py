from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def get_volume_record_from_zapi(self, volume_info, vol_name):
    volume_attributes = self.na_helper.zapi_get_value(volume_info, ['attributes-list', 'volume-attributes'], required=True)
    result = dict(name=vol_name)
    self.get_volume_attributes(volume_attributes, result)
    result['uuid'] = result['instance_uuid'] if result['style_extended'] == 'flexvol' else result['flexgroup_uuid'] if result['style_extended'] is not None and result['style_extended'].startswith('flexgroup') else None
    auto_delete = {}
    self.get_snapshot_auto_delete_attributes(volume_attributes, auto_delete)
    result['snapshot_auto_delete'] = auto_delete
    self.get_efficiency_info(result)
    return result