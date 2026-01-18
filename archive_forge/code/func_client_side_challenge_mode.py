from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def client_side_challenge_mode(self):
    if self._values['mobile_detection'] is None:
        return None
    return self._values['mobile_detection']['client_side_challenge_mode']