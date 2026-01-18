from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
@property
def clone_pools(self):
    if self.want.clone_pools == [] and self.have.clone_pools:
        return self.want.clone_pools
    result = self._diff_complex_items(self.want.clone_pools, self.have.clone_pools)
    return result