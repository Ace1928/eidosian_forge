from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def remove_node_rest(self):
    """
        Remove a node from an existing cluster
        """
    uuid, from_node = self.get_uuid()
    query = {'force': True} if self.parameters.get('force') else None
    dummy, error = rest_generic.delete_async(self.rest_api, 'cluster/nodes', uuid, query, job_timeout=120)
    if error:
        self.module.fail_json(msg='Error removing node with %s: %s' % (from_node, to_native(error)), exception=traceback.format_exc())