from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def enforcement(self):
    to_filter = dict(excess_client_headers=self._values['excess_client_headers'], excess_server_headers=self._values['excess_server_headers'], known_methods=self.known_methods, max_header_count=self.max_header_count, max_header_size=self.max_header_size, max_requests=self.max_requests, oversize_client_headers=self._values['oversize_client_headers'], oversize_server_headers=self._values['oversize_server_headers'], pipeline=self._values['pipeline'], truncated_redirects=self.truncated_redirects, unknown_method=self._values['unknown_method'])
    result = self._filter_params(to_filter)
    if result:
        return result