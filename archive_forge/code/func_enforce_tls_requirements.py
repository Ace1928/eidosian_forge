from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def enforce_tls_requirements(self):
    if self._values['enforce_tls_requirements'] is None:
        return None
    elif self._values['enforce_tls_requirements'] == 'enabled':
        return 'yes'
    return 'no'