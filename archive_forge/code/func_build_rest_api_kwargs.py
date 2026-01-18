from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def build_rest_api_kwargs(self, name):
    if name in ['sp_upgrade', 'sp_version']:
        return {'api': 'cluster/nodes', 'query': {'name': self.parameters['attributes']['node']}, 'fields': self.get_fields(name)}
    if name == 'snapmirror_relationship':
        return {'api': 'snapmirror/relationships', 'query': {'destination.path': self.parameters['attributes']['destination_path']}, 'fields': self.get_fields(name)}
    raise KeyError(name)