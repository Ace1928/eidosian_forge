from __future__ import absolute_import, division, print_function
import re
import sys
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def remove_expired_licenses(self):
    """
        Remove expired licenses
        """
    remove_expired = netapp_utils.zapi.NaElement('license-v2-delete-expired')
    try:
        self.server.invoke_successfully(remove_expired, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error removing expired licenses: %s' % to_native(error), exception=traceback.format_exc())