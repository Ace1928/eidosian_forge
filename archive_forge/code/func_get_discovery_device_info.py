from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
import time
import re
def get_discovery_device_info(self, discovery_id=None, task_id=None):
    """
        Retrieve the information of devices discovered by a specific discovery
        process in the Cisco Catalyst Center. It checks the reachability status of the
        devices periodically until all devices are reachable or until a
        maximum of 3 attempts.

        Parameters:
          - discovery_id: ID of the discovery process to retrieve devices from.
          - task_id: ID of the task associated with the discovery process.

        Returns:
          - result: True if all devices are reachable, False otherwise.
        """
    params = dict(id=discovery_id, task_id=task_id, headers=self.validated_config[0].get('headers'))
    result = False
    count = 0
    while True:
        response = self.dnac_apply['exec'](family='discovery', function='get_discovered_network_devices_by_discovery_id', params=params)
        devices = response.response
        self.log("Retrieved device details using the API 'get_discovered_network_devices_by_discovery_id': {0}".format(str(devices)), 'DEBUG')
        if all((res.get('reachabilityStatus') == 'Success' for res in devices)):
            result = True
            self.log('All devices in the range are reachable', 'INFO')
            break
        elif any((res.get('reachabilityStatus') == 'Success' for res in devices)):
            result = True
            self.log('Some devices in the range are reachable', 'INFO')
            break
        elif all((res.get('reachabilityStatus') != 'Success' for res in devices)):
            result = True
            self.log('All devices are not reachable, but discovery is completed', 'WARNING')
            break
        count += 1
        if count == 3:
            break
        time.sleep(3)
    if not result:
        msg = 'Discovery network device with id {0} has not completed'.format(discovery_id)
        self.log(msg, 'CRITICAL')
        self.module.fail_json(msg=msg)
    self.log('Discovery network device with id {0} got completed'.format(discovery_id), 'INFO')
    self.result.update(dict(discovery_device_info=devices))
    return result