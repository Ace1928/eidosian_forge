from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def destination_to_network(self):
    destination = self._values['destination']
    if destination.startswith('default%'):
        destination = '0.0.0.0%{0}/0'.format(destination.split('%')[1])
    elif destination.startswith('default-inet6%'):
        destination = '::%{0}/0'.format(destination.split('%')[1])
    elif destination.startswith('default-inet6'):
        destination = '::/0'
    elif destination.startswith('default'):
        destination = '0.0.0.0/0'
    return destination