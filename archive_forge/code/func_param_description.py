from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.urls import build_service_uri
from ..module_utils.teem import send_teem
@property
def param_description(self):
    if self._values['parameters'] is None:
        return None
    result = self._values['parameters'].get('description', None)
    return result