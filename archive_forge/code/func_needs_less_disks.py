from __future__ import absolute_import, division, print_function
import json
import logging
import sys
import traceback
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import reduce
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
@property
def needs_less_disks(self):
    if len(self.cache_detail['driveRefs']) > self.disk_count:
        self.debug('needs resize: current disk count %s < requested requested count %s', len(self.cache_detail['driveRefs']), self.disk_count)
        return True