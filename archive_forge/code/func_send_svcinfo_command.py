from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import svc_ssh_argument_spec, get_logger
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_ssh import IBMSVCssh
from ansible.module_utils._text import to_native
def send_svcinfo_command(self):
    info_output = ''
    message = ''
    failed = False
    if self.ssh_client.is_client_connected:
        if not self.command.startswith('svcinfo'):
            failed = True
            message = 'The command must start with svcinfo'
        if self.command.find('|') != -1:
            failed = True
            message = 'Pipe(|) is not supported in command.'
        if self.command.find('-filtervalue') != -1:
            failed = True
            message = "'filtervalue' is not supported in command."
        if not failed:
            new_command = self.modify_command(self.command)
            self.log('Executing CLI command: %s', new_command)
            stdin, stdout, stderr = self.ssh_client.client.exec_command(new_command)
            for line in stdout.readlines():
                info_output += line
            self.log(info_output)
            rc = stdout.channel.recv_exit_status()
            if rc > 0:
                message = stderr.read()
                if len(message) > 0:
                    message = message.decode('utf-8')
                    self.log('Error in executing CLI command: %s', new_command)
                    self.log('%s', message)
                else:
                    message = 'Unknown error'
                self.ssh_client._svc_disconnect()
                self.module.fail_json(msg=message, rc=rc, stdout=info_output)
            self.ssh_client._svc_disconnect()
            self.module.exit_json(msg=message, rc=rc, stdout=info_output, changed=False)
    else:
        message = 'SSH client is not connected'
    self.ssh_client._svc_disconnect()
    self.module.fail_json(msg=message)