from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def remove_portset_ports_rest(self, port, portset_uuid):
    """
        Removes all existing ports from portset
        :return: None
        """
    api = 'protocols/san/portsets/%s/interfaces' % portset_uuid
    dummy, error = rest_generic.delete_async(self.rest_api, api, self.desired_lifs[port]['uuid'])
    if error:
        self.module.fail_json(msg=error)