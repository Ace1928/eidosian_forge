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
def product_changelist(self):
    if 'version_info' not in self._values:
        return None
    if 'Changelist' in self._values['version_info']:
        return int(self._values['version_info']['Changelist'])