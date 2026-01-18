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
def read_volume_from_device(self):
    try:
        resp = self.client.api.get(self.volume_url)
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    except ssl.SSLError:
        return None
    except URLError:
        return None
    if 'code' in response and response['code'] == 400:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)
    return response