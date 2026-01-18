from __future__ import (absolute_import, division, print_function)
import shelve
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.common.text.converters import to_bytes, to_text
def read_shelve(self, shelve_filename, key):
    """
        Read the value of "key" from a shelve file
        """
    d = shelve.open(to_bytes(shelve_filename))
    res = d.get(key, None)
    d.close()
    return res