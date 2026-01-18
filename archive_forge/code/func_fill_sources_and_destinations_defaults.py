from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
from ansible.module_utils._text import to_native
def fill_sources_and_destinations_defaults(self, obj, prop):
    value = obj.get(prop)
    if value is None:
        value = {}
    else:
        value = self.fill_source_and_destination_defaults_inner(value)
    obj[prop] = value
    return obj