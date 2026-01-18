from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def read_dns_cache_setting(self):
    uri = 'https://{0}:{1}/mgmt/tm/sys/db/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], 'dns.cache')
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        return response
    raise F5ModuleError(resp.content)