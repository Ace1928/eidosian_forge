from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def syslog_format(self):
    if self._values['syslog_format'] is None:
        return None
    result = self._values['syslog_format']
    if result == 'syslog':
        result = 'rfc5424'
    if result == 'bsd-syslog':
        result = 'rfc3164'
    return result