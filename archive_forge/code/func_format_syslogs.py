from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def format_syslogs(self, syslogs):
    result = None
    for x in syslogs:
        syslog = ApiParameters(params=x)
        self.syslogs[syslog.name] = x
        if syslog.name == self.want.name:
            result = syslog
        elif syslog.remote_host == self.want.remote_host:
            result = syslog
    if not result:
        return ApiParameters()
    return result