from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def self_allow(self):
    if self._values['self_allow'] is None:
        return None
    to_filter = dict(defaults=self._parse_self_defaults(), all=self._get_all_value())
    result = self._filter_params(to_filter)
    if result:
        return result