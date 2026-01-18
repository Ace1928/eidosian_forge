from __future__ import absolute_import, division, print_function
from ansible_collections.theforeman.foreman.plugins.module_utils.foreman_helper import (
def ensure_host_interfaces(module, entity, interfaces):
    scope = {'host_id': entity['id']}
    current_interfaces = module.list_resource('interfaces', params=scope)
    current_interfaces_ids = {x['id'] for x in current_interfaces}
    expected_interfaces_ids = set()
    for interface in interfaces:
        if 1 == len(current_interfaces) == len(interfaces):
            existing_interface = current_interfaces[0]
        else:
            for possible_identifier in ['identifier', 'name', 'mac']:
                if possible_identifier in interface:
                    unique_identifier = possible_identifier
                    break
            else:
                unique_identifier = None
                warning_msg = 'The provided interface definition has no unique identifier and thus cannot be matched against existing interfaces. This will always create a new interface and might not be the desired behaviour.'
                module.warn(warning_msg)
            existing_interface = next((x for x in current_interfaces if unique_identifier and x.get(unique_identifier) == interface[unique_identifier]), None)
        if 'mac' in interface:
            interface['mac'] = interface['mac'].lower()
        if existing_interface is not None and 'attached_devices' in existing_interface:
            existing_interface['attached_devices'] = existing_interface['attached_devices'].split(',')
        updated_interface = (existing_interface or {}).copy()
        updated_interface.update(interface)
        module.ensure_entity('interfaces', updated_interface, existing_interface, params=scope, state='present', foreman_spec=module.foreman_spec['interfaces_attributes']['foreman_spec'])
        if 'id' in updated_interface:
            expected_interfaces_ids.add(updated_interface['id'])
    for leftover_interface in current_interfaces_ids - expected_interfaces_ids:
        module.ensure_entity('interfaces', {}, {'id': leftover_interface}, params=scope, state='absent', foreman_spec=module.foreman_spec['interfaces_attributes']['foreman_spec'])