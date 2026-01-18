import abc
import copy
from ansible.module_utils.six import raise_from
import importlib
import os
from ansible.module_utils.basic import AnsibleModule
def check_versioned(self, **kwargs):
    """Check that provided arguments are supported by current SDK version

        Returns:
            versioned_result {dict} dictionary of only arguments that are
                                    supported by current SDK version. All others
                                    are dropped.
        """
    versioned_result = {}
    for var_name in kwargs:
        if 'min_ver' in self.argument_spec[var_name] and StrictVersion(self.sdk_version) < self.argument_spec[var_name]['min_ver']:
            continue
        if 'max_ver' in self.argument_spec[var_name] and StrictVersion(self.sdk_version) > self.argument_spec[var_name]['max_ver']:
            continue
        versioned_result.update({var_name: kwargs[var_name]})
    return versioned_result