from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def snapmirror_delete(self):
    """
        Delete SnapMirror relationship at destination cluster
        """
    if self.use_rest:
        return self.snapmirror_delete_rest()
    options = {'destination-location': self.parameters['destination_path']}
    snapmirror_delete = netapp_utils.zapi.NaElement.create_node_with_children('snapmirror-destroy', **options)
    try:
        self.server.invoke_successfully(snapmirror_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        msg = 'Error deleting SnapMirror: %s' % to_native(error)
        if self.previous_errors:
            msg += '.  Previous error(s): %s' % ' -- '.join(self.previous_errors)
        self.module.fail_json(msg=msg, exception=traceback.format_exc())