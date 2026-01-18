from __future__ import absolute_import, division, print_function
from traceback import format_exc
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_ssh import IBMSVCssh
from ansible.module_utils._text import to_native
def is_truststore_exists(self):
    merged_result = {}
    cmd = 'lstruststore -json {0}'.format(self.name)
    stdin, stdout, stderr = self.ssh_client.client.exec_command(cmd)
    result = stdout.read().decode('utf-8')
    if result:
        result = json.loads(result)
    else:
        return merged_result
    rc = stdout.channel.recv_exit_status()
    if rc > 0:
        message = stderr.read().decode('utf-8')
        if message.count('CMMVC5804E') != 1 or message.count('CMMVC6035E') != 1:
            self.log('Error in executing CLI command: %s', cmd)
            self.log('%s', message)
            self.module.fail_json(msg=message)
        else:
            self.log('Expected error: %s', message)
    if isinstance(result, list):
        for d in result:
            merged_result.update(d)
    else:
        merged_result = result
    return merged_result