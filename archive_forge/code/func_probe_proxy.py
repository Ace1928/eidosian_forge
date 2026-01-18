from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def probe_proxy(self, data):
    modify = {}
    if self.proxy_type == 'open_proxy':
        if self.proxy_url:
            if self.proxy_url != data['url']:
                modify['url'] = self.proxy_url
        if self.proxy_port:
            if int(self.proxy_port) != int(data['port']):
                modify['port'] = self.proxy_port
    elif self.proxy_type == 'basic_authentication':
        if self.proxy_url:
            if self.proxy_url != data['url']:
                modify['url'] = self.proxy_url
        if self.proxy_port:
            if self.proxy_port != int(data['port']):
                modify['port'] = self.proxy_port
        if self.proxy_username:
            if self.proxy_username != data['username']:
                modify['username'] = self.proxy_username
        if self.proxy_password:
            modify['password'] = self.proxy_password
    elif self.proxy_type == 'certificate':
        if self.proxy_url:
            if self.proxy_url != data['url']:
                modify['url'] = self.proxy_url
        if self.proxy_port:
            if self.proxy_port != int(data['port']):
                modify['port'] = self.proxy_port
        if self.sslcert:
            modify['sslcert'] = self.sslcert
    return modify