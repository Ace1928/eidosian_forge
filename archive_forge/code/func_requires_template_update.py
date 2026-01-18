from __future__ import (absolute_import, division, print_function)
import time
import ssl
from os import environ
from ansible.module_utils.six import string_types
from ansible.module_utils.basic import AnsibleModule
def requires_template_update(self, current, desired):
    """
        This function will help decide if a template update is required or not
        If a desired key is missing from the current dictionary an update is required
        If the intersection of both dictionaries is not deep equal, an update is required
        Args:
            current: current template as a dictionary
            desired: desired template as a dictionary

        Returns: True if a template update is required
        """
    if not desired:
        return False
    self.cast_template(desired)
    intersection = dict()
    for dkey in desired.keys():
        if dkey in current.keys():
            intersection[dkey] = current[dkey]
        else:
            return True
    return not desired == intersection