from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def log_message_fields(self):
    if self.want.log_message_fields is None:
        return None
    if len(self.want.log_message_fields) == 1:
        if self.have.log_message_fields is None and self.want.log_message_fields[0] in ['', 'none']:
            return None
        if self.have.log_message_fields is not None and self.want.log_message_fields[0] in ['', 'none']:
            return []
    if self.have.log_message_fields is None:
        return self.want.log_message_fields
    if set(self.want.log_message_fields) != set(self.have.log_message_fields):
        return self.want.log_message_fields
    return None