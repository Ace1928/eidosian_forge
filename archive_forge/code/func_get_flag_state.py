from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def get_flag_state(self):
    global_parameters = self._exec(['list_feature_flags'], True)
    for param_item in global_parameters:
        name, state = param_item.split('\t')
        if name == self.name:
            if state == 'enabled':
                return 'enabled'
            return 'disabled'
    return 'unavailable'