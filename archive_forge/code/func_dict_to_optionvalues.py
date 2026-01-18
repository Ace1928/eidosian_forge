from __future__ import (absolute_import, division, print_function)
import time
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def dict_to_optionvalues(self):
    optionvalues = []
    for dictionary in self.params['guestinfo_vars']:
        for key, value in dictionary.items():
            opt = vim.option.OptionValue()
            opt.key, opt.value = ('guestinfo.ic.' + key, value)
            optionvalues.append(opt)
    return optionvalues