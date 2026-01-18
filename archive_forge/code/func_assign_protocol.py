from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
from ansible.module_utils._text import to_native
def assign_protocol(self, patch, origin):
    attribute_map = ['probes', 'inbound_nat_rules', 'inbound_nat_pools', 'load_balancing_rules']
    for attribute in attribute_map:
        properties = getattr(patch, attribute)
        if not properties:
            continue
        references = getattr(origin, attribute) if origin else []
        for item in properties:
            if item.protocol:
                continue
            refs = [x for x in references if to_native(x.name) == item.name]
            ref = refs[0] if len(refs) > 0 else None
            item.protocol = ref.protocol if ref else 'Tcp'
    return patch