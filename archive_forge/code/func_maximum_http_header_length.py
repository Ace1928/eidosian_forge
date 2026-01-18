from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
@property
def maximum_http_header_length(self):
    if 'attributes' in self._values:
        if self._values['attributes'] is None:
            return None
        if 'maximumHttpHeaderLength' in self._values['attributes']:
            if self._values['attributes']['maximumHttpHeaderLength'] == 'any':
                return 'any'
            return int(self._values['attributes']['maximumHttpHeaderLength'])
    if 'header_settings' in self._values:
        if self._values['header_settings'] is None:
            return None
        if 'maximumHttpHeaderLength' in self._values['header_settings']:
            if self._values['header_settings']['maximumHttpHeaderLength'] == 'any':
                return 'any'
            return int(self._values['header_settings']['maximumHttpHeaderLength'])