from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def cluster_image_packages_get_rest(self):
    if not self.use_rest:
        return self.cluster_image_packages_get_zapi()
    api = 'cluster/software/packages'
    records, error = rest_generic.get_0_or_more_records(self.rest_api, api, fields='version')
    return ([record.get('version') for record in records] if records else [], error)