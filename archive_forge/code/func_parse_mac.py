from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import boolean
def parse_mac(netstr):
    return re.search('=(.*?),', netstr).group(1)