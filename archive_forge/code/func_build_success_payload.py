from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def build_success_payload(self, host=None):
    keys = []
    if host:
        result = dict(((key, host[key]) for key in keys))
    else:
        result = dict()
    result['ssid'] = self.ssid
    result['api_url'] = self.url
    return result