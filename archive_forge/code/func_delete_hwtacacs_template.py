from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def delete_hwtacacs_template(self, **kwargs):
    """ Delete hwtacacs template """
    hwtacas_template = kwargs['hwtacas_template']
    module = kwargs['module']
    conf_str = CE_DELETE_HWTACACS_TEMPLATE % hwtacas_template
    xml = self.netconf_set_config(module=module, conf_str=conf_str)
    if '<ok/>' not in xml:
        module.fail_json(msg='Error: Delete hwtacacs template failed.')
    cmds = []
    cmd = 'undo hwtacacs server template %s' % hwtacas_template
    cmds.append(cmd)
    return cmds