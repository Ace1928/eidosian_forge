from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def error_is_fatal(self, error):
    """ a node may not be available during reboot, or the job may be lost """
    if not error:
        return False
    self.rest_api.log_debug('transient_error', error)
    error_messages = ["entry doesn't exist", 'Max retries exceeded with url: /api/cluster/jobs']
    return all((error_message not in error for error_message in error_messages))