from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper
from ansible_collections.community.general.plugins.module_utils.xfconf import xfconf_runner
def process_command_output(self, rc, out, err):
    result = out.rstrip()
    if 'Value is an array with' in result:
        result = result.split('\n')
        result.pop(0)
        result.pop(0)
        self.vars.is_array = True
    return result