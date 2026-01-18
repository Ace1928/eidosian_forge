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
def lldp(self):
    if self._values['lldp'] is None:
        return None
    to_filter = dict(enabled=self.enabled, max_neighbors_per_port=self._values['lldp'].get('maxNeighborsPerPort', None), reinit_delay=self._values['lldp'].get('reinitDelay', None), tx_delay=self._values['lldp'].get('txDelay', None), tx_hold=self._values['lldp'].get('txHold', None), tx_interval=self._values['lldp'].get('txInterval', None))
    result = self._filter_params(to_filter)
    if result:
        return result