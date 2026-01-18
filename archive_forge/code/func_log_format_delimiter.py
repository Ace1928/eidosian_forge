from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def log_format_delimiter(self):
    if self._values['log_format_delimiter'] is None:
        return None
    if len(self._values['log_format_delimiter']) > 31:
        raise F5ModuleError('The maximum length of delimiter is 31 characters.')
    if '$' in self._values['log_format_delimiter']:
        raise F5ModuleError("Cannot use '$' character as a part of delimiter.")
    return self._values['log_format_delimiter']