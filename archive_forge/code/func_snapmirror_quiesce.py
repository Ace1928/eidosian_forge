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
def snapmirror_quiesce(self):
    """
        Quiesce SnapMirror relationship - disable all future transfers to this destination
        """
    if self.use_rest:
        return self.snapmirror_quiesce_rest()
    options = {'destination-location': self.parameters['destination_path']}
    snapmirror_quiesce = netapp_utils.zapi.NaElement.create_node_with_children('snapmirror-quiesce', **options)
    try:
        self.server.invoke_successfully(snapmirror_quiesce, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error quiescing SnapMirror: %s' % to_native(error), exception=traceback.format_exc())
    self.wait_for_quiesced_status()