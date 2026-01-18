from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from datetime import datetime
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def lb_method(self):
    lb_method = self._values['lb_method']
    if lb_method is None:
        return None
    spec = ArgumentSpec()
    if lb_method not in spec.lb_choice:
        raise F5ModuleError('Provided lb_method is unknown')
    return lb_method