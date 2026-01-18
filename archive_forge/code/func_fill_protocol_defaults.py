from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
from ansible.module_utils._text import to_native
def fill_protocol_defaults(self, obj):
    if obj.get('protocol') is None:
        obj['protocol'] = 'tcp'
    return obj