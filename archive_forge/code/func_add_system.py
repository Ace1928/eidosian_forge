from __future__ import absolute_import, division, print_function
import json
import threading
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def add_system(self, system):
    """Add basic storage system definition to the web services proxy."""
    self.set_password(system)
    body = {'id': system['ssid'], 'controllerAddresses': system['controller_addresses'], 'password': system['password']}
    if system['accept_certificate']:
        body.update({'acceptCertificate': system['accept_certificate']})
    if system['meta_tags']:
        body.update({'metaTags': system['meta_tags']})
    try:
        rc, storage_system = self.request('storage-systems', method='POST', data=body)
    except Exception as error:
        self.module.warn('Failed to add storage system. Array [%s]. Error [%s]' % (system['ssid'], to_native(error)))
        return
    for retries in range(5):
        sleep(1)
        try:
            rc, storage_system = self.request('storage-systems/%s/validatePassword' % system['ssid'], method='POST')
            break
        except Exception as error:
            continue
    else:
        self.module.warn('Failed to validate password status. Array [%s]. Error [%s]' % (system['ssid'], to_native(error)))