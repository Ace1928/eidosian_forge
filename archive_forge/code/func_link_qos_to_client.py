from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def link_qos_to_client(self):
    result = self.transform_link_qos('link_qos_to_client')
    if result == -1:
        raise F5ModuleError('link_qos_to_client must be between 0 and 7')
    return result