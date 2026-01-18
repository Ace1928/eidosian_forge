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
def allow_any_ios_package(self):
    if self._values['mobile_detection'] is None:
        return None
    result = flatten_boolean(self._values['mobile_detection']['allow_any_ios_package'])
    if result == 'yes':
        return 'true'
    if result == 'no':
        return 'false'
    return result