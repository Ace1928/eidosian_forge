from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_disk_options(self, modify):
    """
        Modifies a nodes disk options
        :return: None
        """
    api = 'private/cli/storage/disk/option'
    query = {'node': self.parameters['node']}
    dummy, error = rest_generic.patch_async(self.rest_api, api, None, modify, query)
    if error:
        self.module.fail_json(msg='Error %s' % error)