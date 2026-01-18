from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def port_block_allocation(self):
    to_filter = dict(blockIdleTimeout=self._values['pba_block_idle_timeout'], blockLifetime=self._values['pba_block_lifetime'], blockSize=self._values['pba_block_size'], clientBlockLimit=self._values['pba_client_block_limit'], zombieTimeout=self._values['pba_zombie_timeout'])
    result = self._filter_params(to_filter)
    if result:
        return result