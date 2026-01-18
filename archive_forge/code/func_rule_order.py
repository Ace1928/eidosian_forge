from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_complex_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def rule_order(self):
    if self._values['rule_order'] is None:
        return None
    if 0 < self._values['rule_order'] > 4294967295:
        raise F5ModuleError('Specified number is out of valid range, correct range is between 0 and 4294967295.')
    return self._values['rule_order']