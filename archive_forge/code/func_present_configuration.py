from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def present_configuration(self):
    configuration = self.get_configuration()
    args = self._get_common_configuration_args()
    args['value'] = self.get_value()
    empty_value = args['value'] in [None, ''] and 'value' not in configuration
    if self.has_changed(args, configuration, ['value']) and (not empty_value):
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('updateConfiguration', **args)
            configuration = res['configuration']
    return configuration