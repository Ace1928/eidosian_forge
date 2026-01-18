from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
import time
import re
def lookup_discovery_by_range_via_name(self):
    """
        Retrieve a specific discovery by name from a range of
        discoveries in the Cisco Catalyst Center.

        Returns:
          - discovery: The discovery with the specified name from the range
                       of discoveries. If no matching discovery is found, it
                       returns None.
        """
    start_index = self.validated_config[0].get('start_index')
    records_to_return = self.validated_config[0].get('records_to_return')
    response = {'response': []}
    if records_to_return > 500:
        num_intervals = records_to_return // 500
        for num in range(0, num_intervals + 1):
            params = dict(start_index=1 + num * 500, records_to_return=500, headers=self.validated_config[0].get('headers'))
            response_part = self.dnac_apply['exec'](family='discovery', function='get_discoveries_by_range', params=params)
            response['response'].extend(response_part['response'])
    else:
        params = dict(start_index=self.validated_config[0].get('start_index'), records_to_return=self.validated_config[0].get('records_to_return'), headers=self.validated_config[0].get('headers'))
        response = self.dnac_apply['exec'](family='discovery', function='get_discoveries_by_range', params=params)
    self.log('Response of the get discoveries via range API is {0}'.format(str(response)), 'DEBUG')
    return next(filter(lambda x: x['name'] == self.validated_config[0].get('discovery_name'), response.get('response')), None)