from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_peer_cluster_name_rest(self):
    """
        Get local cluster name
        :return: cluster name
        """
    api = 'cluster'
    options = {'fields': 'name'}
    restapi = self.dst_rest_api if self.is_remote_peer() else self.rest_api
    record, error = rest_generic.get_one_record(restapi, api, options)
    if error:
        self.module.fail_json(msg='Error fetching peer cluster name for peer vserver %s: %s' % (self.parameters['peer_vserver'], error))
    if record is not None:
        return record.get('name')
    return None