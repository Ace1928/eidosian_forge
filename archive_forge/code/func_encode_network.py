from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from datetime import datetime
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip_interface
from ..module_utils.teem import send_teem
def encode_network(self, key, prefixlen, value):
    return 'network {0} prefixlen {1} {2} {3}'.format(str(key), str(prefixlen), self._separator, str(value))