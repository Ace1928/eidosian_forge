from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
@property
def redirect_virtual_server(self):
    result = dict()
    result['ltm:virtual:40e8c4a6f542'] = [dict(parameters=dict(name='default_redirect_vs', destinationAddress=self.redirect_virtual['address'], mask=self.redirect_virtual['netmask'], destinationPort=self.redirect_virtual['port']), subcollectionResources=self.redirect_profiles)]
    return result