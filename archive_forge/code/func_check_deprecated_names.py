import abc
import copy
from ansible.module_utils.six import raise_from
import importlib
import os
from ansible.module_utils.basic import AnsibleModule
def check_deprecated_names(self):
    """Check deprecated module names if `deprecated_names` variable is set.
        """
    new_module_name = OVERRIDES.get(self.module_name)
    if self.module_name in self.deprecated_names and new_module_name:
        self.ansible.deprecate("The '%s' module has been renamed to '%s' in openstack collection: openstack.cloud.%s" % (self.module_name, new_module_name, new_module_name), version='3.0.0', collection_name='openstack.cloud')