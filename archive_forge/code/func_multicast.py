from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def multicast(self):
    if self._values['multicast'] is None:
        return None
    to_filter = dict(max_pending_packets=self._values['multicast'].get('maxPendingPackets', None), max_pending_routes=self._values['multicast'].get('maxPendingRoutes', None), rate_limit=flatten_boolean(self._values['multicast'].get('rateLimit', None)), route_lookup_timeout=self._values['multicast'].get('routeLookupTimeout', None))
    result = self._filter_params(to_filter)
    if result:
        return result