from __future__ import absolute_import, division, print_function
import time
import ssl
from datetime import datetime
from ansible.module_utils.six.moves.urllib.error import URLError
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def software_installation_exists(self):
    if self.volume_url is None:
        return False
    resp = self.client.api.get(self.volume_url)
    try:
        resp_json = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp_json.get('status') and resp_json.get('status') == 404:
        return False
    same_version = self.want.version == resp_json.get('version', None)
    same_build = self.want.build == resp_json.get('build', None)
    if not same_build or not same_version:
        return False
    return True