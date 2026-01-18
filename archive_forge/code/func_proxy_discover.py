from __future__ import absolute_import, division, print_function
import json
import multiprocessing
import threading
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import request
from ansible.module_utils._text import to_native
def proxy_discover(self):
    """Search for array using it's chassis serial from web services proxy."""
    self.verify_proxy_service()
    subnet = ipaddress.ip_network(u'%s' % self.subnet_mask)
    try:
        rc, request_id = request(self.proxy_url + 'discovery', method='POST', validate_certs=self.proxy_validate_certs, force_basic_auth=True, url_username=self.proxy_username, url_password=self.proxy_password, data=json.dumps({'startIP': str(subnet[0]), 'endIP': str(subnet[-1]), 'connectionTimeout': self.DEFAULT_CONNECTION_TIMEOUT_SEC}))
        try:
            for iteration in range(self.DEFAULT_DISCOVERY_TIMEOUT_SEC):
                rc, discovered_systems = request(self.proxy_url + 'discovery?requestId=%s' % request_id['requestId'], validate_certs=self.proxy_validate_certs, force_basic_auth=True, url_username=self.proxy_username, url_password=self.proxy_password)
                if not discovered_systems['discoverProcessRunning']:
                    thread_pool = []
                    for discovered_system in discovered_systems['storageSystems']:
                        addresses = []
                        for controller in discovered_system['controllers']:
                            addresses.extend(controller['ipAddresses'])
                        if 'https' in discovered_system['supportedManagementPorts'] and self.prefer_embedded:
                            thread = threading.Thread(target=self.test_systems_found, args=(self.systems_found, discovered_system['serialNumber'], discovered_system['label'], addresses))
                            thread_pool.append(thread)
                            thread.start()
                        else:
                            self.systems_found.update({discovered_system['serialNumber']: {'api_urls': [self.proxy_url], 'label': discovered_system['label'], 'addresses': addresses, 'proxy_ssid': '', 'proxy_required': True}})
                    for thread in thread_pool:
                        thread.join()
                    break
                sleep(1)
            else:
                self.module.fail_json(msg='Timeout waiting for array discovery process. Subnet [%s]' % self.subnet_mask)
        except Exception as error:
            self.module.fail_json(msg='Failed to get the discovery results. Error [%s].' % to_native(error))
    except Exception as error:
        self.module.fail_json(msg='Failed to initiate array discovery. Error [%s].' % to_native(error))