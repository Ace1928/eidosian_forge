from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def initiate_cloud_callhome(self):
    msg = ''
    attempts = 0
    limit_reached = False
    active_status = False
    self.manage_proxy_server()
    self.update_email_data()
    lsdata = self.get_existing_cloud_callhome_data()
    if lsdata['status'] == 'enabled':
        self.test_connection_cloud_callhome()
    else:
        self.enable_cloud_callhome()
        while not active_status:
            attempts += 1
            if attempts > 10:
                limit_reached = True
                break
            lsdata = self.get_existing_cloud_callhome_data()
            if lsdata['status'] == 'enabled':
                active_status = True
            time.sleep(2)
        if limit_reached:
            msg = 'Callhome with Cloud is enabled. Please check connection to proxy.'
            self.changed = True
            return msg
        if active_status:
            self.test_connection_cloud_callhome()
    msg = 'Callhome with Cloud enabled successfully.'
    self.changed = True
    return msg