from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
@property
def default_host_type(self):
    """Return the default host type index."""
    try:
        rc, default_index = self.request('storage-systems/%s/graph/xpath-filter?query=/sa/defaultHostTypeIndex' % self.ssid)
        return default_index[0]
    except Exception as error:
        self.module.fail_json(msg='Failed to retrieve default host type index')