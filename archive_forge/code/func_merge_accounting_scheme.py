from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def merge_accounting_scheme(self, **kwargs):
    """ Merge scheme of accounting """
    acct_scheme_name = kwargs['acct_scheme_name']
    accounting_mode = kwargs['accounting_mode']
    module = kwargs['module']
    conf_str = CE_MERGE_ACCOUNTING_SCHEME % (acct_scheme_name, accounting_mode)
    xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in xml:
        module.fail_json(msg='Error: Merge accounting scheme failed.')
    cmds = []
    cmd = 'accounting-scheme %s' % acct_scheme_name
    cmds.append(cmd)
    cmd = 'accounting-mode %s' % accounting_mode
    cmds.append(cmd)
    return cmds