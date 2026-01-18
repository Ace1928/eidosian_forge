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
def get_snapshot_auto_delete_attributes(self, volume_attributes, result):
    attrs = dict(commitment=dict(key_list=['volume-snapshot-autodelete-attributes', 'commitment']), defer_delete=dict(key_list=['volume-snapshot-autodelete-attributes', 'defer-delete']), delete_order=dict(key_list=['volume-snapshot-autodelete-attributes', 'delete-order']), destroy_list=dict(key_list=['volume-snapshot-autodelete-attributes', 'destroy-list']), is_autodelete_enabled=dict(key_list=['volume-snapshot-autodelete-attributes', 'is-autodelete-enabled'], convert_to=bool), prefix=dict(key_list=['volume-snapshot-autodelete-attributes', 'prefix']), target_free_space=dict(key_list=['volume-snapshot-autodelete-attributes', 'target-free-space'], convert_to=int), trigger=dict(key_list=['volume-snapshot-autodelete-attributes', 'trigger']))
    self.na_helper.zapi_get_attrs(volume_attributes, attrs, result)
    if result['is_autodelete_enabled'] is not None:
        result['state'] = 'on' if result['is_autodelete_enabled'] else 'off'
        del result['is_autodelete_enabled']