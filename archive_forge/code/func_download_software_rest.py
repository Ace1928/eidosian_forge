from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def download_software_rest(self):
    body = {'url': self.parameters['package_url']}
    for attr in ('username', 'password'):
        value = self.parameters.get('server_%s' % attr)
        if value:
            body[attr] = value
    api = 'cluster/software/download'
    message, error = rest_generic.post_async(self.rest_api, api, body, job_timeout=self.parameters.get('time_out', 180), timeout=0)
    if error:
        self.module.fail_json(msg='Error downloading software: %s' % error)
    return message