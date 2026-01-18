from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def is_update_complete_rest(self):
    state, error = self.cluster_image_get_rest('state', fail_on_error=False)
    if error:
        return (None, None, error)
    return (state in ['paused_by_user', 'paused_on_error', 'completed', 'canceled', 'failed'], state, error)