from __future__ import absolute_import, division, print_function
import time
import json
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.basic import to_text
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import BASE_HEADERS
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGBaseException
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGRCommon
from datetime import datetime
def get_lock_info(self, adom=None):
    """
        Gets ADOM lock info so it can be displayed with the error messages. Or if determined to be locked by ansible
        for some reason, then unlock it.
        """
    url = '/dvmdb/adom/root/workspace/lockinfo'
    if adom and adom != 'root':
        if adom.lower() == 'global':
            url = '/dvmdb/global/workspace/lockinfo'
        else:
            url = '/dvmdb/adom/{adom}/workspace/lockinfo/'.format(adom=adom)
    rc, resp_obj = self.send_request('get', self._tools.format_request('get', url))
    return (rc, resp_obj)