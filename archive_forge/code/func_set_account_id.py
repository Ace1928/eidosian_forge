from __future__ import absolute_import, division, print_function
import traceback
import uuid
import time
import base64
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def set_account_id(self):
    if self.parameters.get('account_id') is None:
        response, error = self.na_helper.get_or_create_account(self.rest_api)
        if error is not None:
            return error
        self.parameters['account_id'] = response
    return None