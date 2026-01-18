from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def process_curd(self, argument_specs=None):
    self.metadata = argument_specs
    module_name = self.module_level2_name
    params = self.module.params
    track = [module_name]
    if not params.get('bypass_validation', False):
        self.check_versioning_mismatch(track, argument_specs.get(module_name, None), params.get(module_name, None))
    if self.module_primary_key and isinstance(params.get(module_name, None), dict):
        mvalue = ''
        if self.module_primary_key.startswith('complex:'):
            mvalue_exec_string = self.module_primary_key[len('complex:'):]
            mvalue_exec_string = mvalue_exec_string.replace('{{module}}', 'self.module.params[self.module_level2_name]')
            mvalue = eval(mvalue_exec_string)
        else:
            mvalue = params[module_name][self.module_primary_key]
        self.do_exit(self._process_with_mkey(mvalue))
    else:
        self.do_exit(self._process_without_mkey())