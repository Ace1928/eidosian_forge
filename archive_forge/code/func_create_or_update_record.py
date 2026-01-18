from __future__ import absolute_import, division, print_function
import traceback
from binascii import Error as binascii_error
from socket import error as socket_error
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def create_or_update_record(self):
    result = {'changed': False, 'failed': False}
    exists = self.record_exists()
    if exists in [0, 2]:
        if self.module.check_mode:
            self.module.exit_json(changed=True)
        if exists == 0:
            self.dns_rc = self.create_record()
            if self.dns_rc != 0:
                result['msg'] = 'Failed to create DNS record (rc: %d)' % self.dns_rc
        elif exists == 2:
            self.dns_rc = self.modify_record()
            if self.dns_rc != 0:
                result['msg'] = 'Failed to update DNS record (rc: %d)' % self.dns_rc
        if self.dns_rc != 0:
            result['failed'] = True
        else:
            result['changed'] = True
    else:
        result['changed'] = False
    return result