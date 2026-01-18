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
def snapmirror_release(self):
    """
        Release SnapMirror relationship from source cluster
        """
    if self.use_rest:
        return
    options = {'destination-location': self.parameters['destination_path'], 'relationship-info-only': self.na_helper.get_value_for_bool(False, self.parameters['relationship_info_only'])}
    snapmirror_release = netapp_utils.zapi.NaElement.create_node_with_children('snapmirror-release', **options)
    try:
        self.source_server.invoke_successfully(snapmirror_release, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error releasing SnapMirror relationship: %s' % to_native(error), exception=traceback.format_exc())