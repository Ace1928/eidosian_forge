from __future__ import (absolute_import, division, print_function)
import time
import ssl
from os import environ
from ansible.module_utils.six import string_types
from ansible.module_utils.basic import AnsibleModule
def cast_template(self, template):
    """
        OpenNebula handles all template elements as strings
        At some point there is a cast being performed on types provided by the user
        This function mimics that transformation so that required template updates are detected properly
        additionally an array will be converted to a comma separated list,
        which works for labels and hopefully for something more.

        Args:
            template: the template to transform

        Returns: the transformed template with data casts applied.
        """
    for key in template:
        value = template[key]
        if isinstance(value, dict):
            self.cast_template(template[key])
        elif isinstance(value, list):
            template[key] = ', '.join(value)
        elif not isinstance(value, string_types):
            template[key] = str(value)