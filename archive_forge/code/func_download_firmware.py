from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def download_firmware(self):
    if self.use_rest:
        return self.download_software_rest()
    ' calls the system-cli ZAPI as there is no ZAPI for this feature '
    msg = MSGS['dl_completed']
    command = ['storage', 'firmware', 'download', '-node', self.parameters['node'] if self.parameters.get('node') else '*', '-package-url', self.parameters['package_url']]
    command_obj = netapp_utils.zapi.NaElement('system-cli')
    args_obj = netapp_utils.zapi.NaElement('args')
    for arg in command:
        args_obj.add_new_child('arg', arg)
    command_obj.add_child_elem(args_obj)
    command_obj.add_new_child('priv', 'advanced')
    output = None
    try:
        output = self.server.invoke_successfully(command_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        try:
            err_num = int(error.code)
        except ValueError:
            err_num = -1
        if err_num == 60:
            msg = MSGS['dl_completed_slowly']
        elif err_num == 502 and (not self.parameters['fail_on_502_error']):
            msg = MSGS['dl_in_progress']
        else:
            self.module.fail_json(msg='Error running command %s: %s' % (command, to_native(error)), exception=traceback.format_exc())
    except netapp_utils.zapi.etree.XMLSyntaxError as error:
        self.module.fail_json(msg='Error decoding output from command %s: %s' % (command, to_native(error)), exception=traceback.format_exc())
    if output is not None:
        status = output.get_attr('status')
        cli_output = output.get_child_content('cli-output')
        if status is None or status != 'passed' or cli_output is None or (cli_output == ''):
            if status is None:
                extra_info = "'status' attribute missing"
            elif status != 'passed':
                extra_info = "check 'status' value"
            else:
                extra_info = 'check console permissions'
            self.module.fail_json(msg='unable to download package from %s: %s.  Received: %s' % (self.parameters['package_url'], extra_info, output.to_string()))
        if cli_output is not None:
            if cli_output.startswith('Error:') or 'Failed to download package from' in cli_output:
                self.module.fail_json(msg='failed to download package from %s: %s' % (self.parameters['package_url'], cli_output))
            msg += '  Extra info: %s' % cli_output
    return msg