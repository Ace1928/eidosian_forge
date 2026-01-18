from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def generate_hardware_dict(module, blade, api_version):
    hw_info = {'modules': {}, 'ethernet': {}, 'mgmt_ports': {}, 'fans': {}, 'bays': {}, 'controllers': {}, 'blades': {}, 'chassis': {}, 'power': {}, 'switch': {}}
    blade = get_system(module)
    components = list(blade.get_hardware(filter="type='fm'").items)
    for component in range(0, len(components)):
        component_name = components[component].name
        hw_info['modules'][component_name] = {'slot': components[component].slot, 'status': components[component].status, 'serial': components[component].serial, 'model': components[component].model, 'identify': components[component].identify_enabled}
        if PART_NUMBER_API_VERSION in api_version:
            hw_info['modules'][component_name]['part_number'] = components[component].part_number
    components = list(blade.get_hardware(filter="type='eth'").items)
    for component in range(0, len(components)):
        component_name = components[component].name
        hw_info['ethernet'][component_name] = {'slot': components[component].slot, 'status': components[component].status, 'serial': components[component].serial, 'model': components[component].model, 'speed': components[component].speed}
        if PART_NUMBER_API_VERSION in api_version:
            hw_info['ethernet'][component_name]['part_number'] = components[component].part_number
    components = list(blade.get_hardware(filter="type='mgmt_port'").items)
    for component in range(0, len(components)):
        component_name = components[component].name
        hw_info['mgmt_ports'][component_name] = {'slot': components[component].slot, 'status': components[component].status, 'serial': components[component].serial, 'model': components[component].model, 'speed': components[component].speed}
        if PART_NUMBER_API_VERSION in api_version:
            hw_info['mgmt_ports'][component_name]['part_number'] = components[component].part_number
    components = list(blade.get_hardware(filter="type='fan'").items)
    for component in range(0, len(components)):
        component_name = components[component].name
        hw_info['fans'][component_name] = {'slot': components[component].slot, 'status': components[component].status, 'identify': components[component].identify_enabled}
        if PART_NUMBER_API_VERSION in api_version:
            hw_info['fans'][component_name]['part_number'] = components[component].part_number
    components = list(blade.get_hardware(filter="type='fb'").items)
    for component in range(0, len(components)):
        component_name = components[component].name
        hw_info['blades'][component_name] = {'slot': components[component].slot, 'status': components[component].status, 'serial': components[component].serial, 'model': components[component].model, 'temperature': components[component].temperature, 'identify': components[component].identify_enabled}
        if PART_NUMBER_API_VERSION in api_version:
            hw_info['blades'][component_name]['part_number'] = components[component].part_number
    components = list(blade.get_hardware(filter="type='pwr'").items)
    for component in range(0, len(components)):
        component_name = components[component].name
        hw_info['power'][component_name] = {'slot': components[component].slot, 'status': components[component].status, 'serial': components[component].serial, 'model': components[component].model}
        if PART_NUMBER_API_VERSION in api_version:
            hw_info['power'][component_name]['part_number'] = components[component].part_number
    components = list(blade.get_hardware(filter="type='xfm'").items)
    for component in range(0, len(components)):
        component_name = components[component].name
        hw_info['switch'][component_name] = {'slot': components[component].slot, 'status': components[component].status, 'serial': components[component].serial, 'model': components[component].model}
        if PART_NUMBER_API_VERSION in api_version:
            hw_info['switch'][component_name]['part_number'] = components[component].part_number
    components = list(blade.get_hardware(filter="type='ch'").items)
    for component in range(0, len(components)):
        component_name = components[component].name
        hw_info['chassis'][component_name] = {'slot': components[component].slot, 'index': components[component].index, 'status': components[component].status, 'serial': components[component].serial, 'model': components[component].model}
        if PART_NUMBER_API_VERSION in api_version:
            hw_info['chassis'][component_name]['part_number'] = components[component].part_number
    components = list(blade.get_hardware(filter="type='bay'").items)
    for component in range(0, len(components)):
        component_name = components[component].name
        hw_info['bays'][component_name] = {'slot': components[component].slot, 'index': components[component].index, 'status': components[component].status, 'serial': components[component].serial, 'model': components[component].model, 'identify': components[component].identify_enabled}
        if PART_NUMBER_API_VERSION in api_version:
            hw_info['bay'][component_name]['part_number'] = components[component].part_number
    return hw_info