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
def has_fasthttp_profiles(self):
    """Check if ``fasthttp`` profile is in API profiles

        This method is used to determine the server type when doing comparisons
        in the Difference class.

        Returns:
             bool: True if server has ``fasthttp`` profiles. False otherwise.
        """
    if self.profiles is None:
        return None
    current = self._read_current_fasthttp_profiles_from_device()
    result = [x['name'] for x in self.profiles if x['name'] in current]
    if len(result) > 0:
        return True
    return False