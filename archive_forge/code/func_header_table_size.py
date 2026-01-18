from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def header_table_size(self):
    header = self._values['header_table_size']
    if header is None:
        return None
    if header < 0 or header > 65535:
        raise F5ModuleError('Header Table Size value must be between 0 and 65535')
    return self._values['header_table_size']