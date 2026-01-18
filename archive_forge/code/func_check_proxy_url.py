from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def check_proxy_url(self, current):
    port = None
    if current.get('proxy_url') is not None:
        port = current['proxy_url'].rstrip('/').split(':')[-1]
    pos = self.parameters['proxy_url'].rstrip('/').rfind(':')
    if self.parameters['proxy_url'][pos + 1] == '/':
        if port is not None and port != '':
            self.parameters['proxy_url'] = '%s:%s' % (self.parameters['proxy_url'].rstrip('/'), port)