from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
@property
def ss_vol_needs_update(self):
    if self.ss_vol['fullWarnThreshold'] != self.full_threshold:
        return True
    else:
        return False