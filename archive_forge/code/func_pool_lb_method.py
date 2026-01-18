from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def pool_lb_method(self):
    result = dict(lb_method=self._values['pool_lb_method'], pool_lb_method=self._values['pool_lb_method'])
    return result