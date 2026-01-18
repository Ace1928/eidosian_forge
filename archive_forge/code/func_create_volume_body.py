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
def create_volume_body(self):
    """Create body for nas template"""
    nas = dict(application_components=[self.create_nas_application_component()])
    value = self.na_helper.safe_get(self.parameters, ['snapshot_policy'])
    if value is not None:
        nas['protection_type'] = {'local_policy': value}
    for attr in ('nfs_access', 'cifs_access'):
        value = self.na_helper.safe_get(self.parameters, ['nas_application_template', attr])
        if value is not None:
            value = self.na_helper.filter_out_none_entries(value)
            if value:
                nas[attr] = value
    for attr in ('exclude_aggregates',):
        values = self.na_helper.safe_get(self.parameters, ['nas_application_template', attr])
        if values:
            nas[attr] = [dict(name=name) for name in values]
    return self.rest_app.create_application_body('nas', nas, smart_container=True)