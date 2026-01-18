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
@property
def gather_subset(self):
    if isinstance(self._values['gather_subset'], string_types):
        self._values['gather_subset'] = [self._values['gather_subset']]
    elif not isinstance(self._values['gather_subset'], list):
        raise F5ModuleError('The specified gather_subset must be a list.')
    tmp = list(set(self._values['gather_subset']))
    tmp.sort()
    self._values['gather_subset'] = tmp
    return self._values['gather_subset']