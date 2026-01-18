from __future__ import absolute_import, division, print_function
import collections
import os
from copy import deepcopy
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import Version
def convert_key_to_base64(self):
    """IOS-XR only accepts base64 decoded files, this converts the public key to a temp file."""
    if self._module.params['aggregate']:
        name = 'aggregate'
    else:
        name = self._module.params['name']
    if self._module.params['public_key_contents']:
        key = self._module.params['public_key_contents']
    elif self._module.params['public_key']:
        readfile = open(self._module.params['public_key'], 'r')
        key = readfile.read()
    splitfile = key.split()[1]
    base64key = b64decode(splitfile)
    base64file = open('/tmp/publickey_%s.b64' % name, 'wb')
    base64file.write(base64key)
    base64file.close()
    return '/tmp/publickey_%s.b64' % name