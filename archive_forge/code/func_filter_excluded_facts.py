from __future__ import absolute_import, division, print_function
import copy
import datetime
import traceback
import math
import re
from ansible.module_utils.basic import (
from ansible.module_utils.six import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.teem import send_teem
def filter_excluded_facts(self):
    exclude = [x[1:] for x in self.want.gather_subset if x[0] == '!']
    include = [x for x in self.want.gather_subset if x[0] != '!']
    result = [x for x in include if x not in exclude]
    return result