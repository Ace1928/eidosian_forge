from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def phase1_key(self):
    if self._values['phase1_key'] is None:
        return None
    if self._values['phase1_key'] in ['', 'none']:
        return ''
    return fq_name(self.partition, self._values['phase1_key'])