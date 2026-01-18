from __future__ import (absolute_import, division, print_function)
import re
import json
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_bytes, to_text
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.common._collections_compat import Mapping
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, dumps
from ansible.plugins.cliconf import CliconfBase
def get_default_flag(self):
    return 'detail'