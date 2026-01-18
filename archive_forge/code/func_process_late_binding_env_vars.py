from __future__ import absolute_import, division, print_function
import os
def process_late_binding_env_vars(self, option_vars):
    """looks through a set of options, and if empty/None, looks for a value in specified env vars, or sets an optional default"""
    for opt, config in option_vars.items():
        for env in config['env']:
            if self._options.has_option(opt) and self._options.get_option(opt) is None:
                self._options.set_option(opt, os.environ.get(env))
        if 'default' in config and self._options.has_option(opt) and (self._options.get_option(opt) is None):
            self._options.set_option(opt, config['default'])
        if 'required' in config and self._options.get_option_default(opt) is None:
            raise HashiVaultValueError('Required option %s was not set.' % opt)