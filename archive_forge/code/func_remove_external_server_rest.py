from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def remove_external_server_rest(self, server):
    api = 'security/key-managers/%s/key-servers' % self.uuid
    dummy, error = rest_generic.delete_async(self.rest_api, api, server)
    if error:
        self.module.fail_json(msg='Error removing external key server %s: %s' % (server, error))