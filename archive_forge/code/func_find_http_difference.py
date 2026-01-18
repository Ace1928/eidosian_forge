from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.api import (
def find_http_difference(self, key, resource, param):
    is_different = False
    if param != resource['http'][key]:
        is_different = True
    return is_different