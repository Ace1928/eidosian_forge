from __future__ import absolute_import, division, print_function
import time
import json
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.basic import to_text
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import BASE_HEADERS
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGBaseException
from ansible_collections.fortinet.fortimanager.plugins.module_utils.common import FMGRCommon
from datetime import datetime
def commit_changes(self, adom=None, aux=False):
    """
        Commits changes to an ADOM
        """
    if adom:
        if aux:
            url = '/pm/config/adom/{adom}/workspace/commit'.format(adom=adom)
        elif adom.lower() == 'global':
            url = '/dvmdb/global/workspace/commit/'
        else:
            url = '/dvmdb/adom/{adom}/workspace/commit'.format(adom=adom)
    else:
        url = '/dvmdb/adom/root/workspace/commit'
    return self.send_request('exec', self._tools.format_request('exec', url))