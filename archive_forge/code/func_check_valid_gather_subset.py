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
def check_valid_gather_subset(self, includes):
    """Check that the specified subset is valid

        The ``gather_subset`` parameter is specified as a "raw" field which means that
        any Python type could technically be provided

        :param includes:
        :return:
        """
    keys = self.managers.keys()
    result = []
    for x in includes:
        if x not in keys:
            if x[0] == '!':
                if x[1:] not in keys:
                    result.append(x)
            else:
                result.append(x)
    return result