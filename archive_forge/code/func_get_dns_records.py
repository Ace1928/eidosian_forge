from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_url
def get_dns_records(self, zone_name=None, type=None, record=None, value=''):
    if not zone_name:
        zone_name = self.zone
    if not type:
        type = self.type
    if not record:
        record = self.record
    if not value and value is not None:
        value = self.value
    zone_id = self._get_zone_id()
    api_call = '/zones/{0}/dns_records'.format(zone_id)
    query = {}
    if type:
        query['type'] = type
    if record:
        query['name'] = record
    if value:
        query['content'] = value
    if query:
        api_call += '?' + urlencode(query)
    records, status = self._cf_api_call(api_call)
    return records