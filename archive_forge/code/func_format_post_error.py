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
def format_post_error(self, error, body):
    if 'The system received a licensing request with an invalid digital signature.' in error:
        key = self.get_key(error, body)
        if key and "'statusResp'" in key:
            error = 'Original NLF contents were modified by Ansible.  Make sure to use the string filter.  REST error: %s' % error
    return error