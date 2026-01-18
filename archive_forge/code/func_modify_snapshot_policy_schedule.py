from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_snapshot_policy_schedule(self, options, zapi):
    """
        Add, modify or remove a schedule to/from a snapshot policy
        """
    snapshot_obj = netapp_utils.zapi.NaElement.create_node_with_children(zapi, **options)
    try:
        self.server.invoke_successfully(snapshot_obj, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying snapshot policy schedule %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())