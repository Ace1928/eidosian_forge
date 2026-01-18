from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes
def set_interface_option(module, lines, iface, option, raw_value, state, address_family=None):
    value = str(raw_value)
    changed = False
    iface_lines = [item for item in lines if 'iface' in item and item['iface'] == iface]
    if address_family is not None:
        iface_lines = [item for item in iface_lines if 'address_family' in item and item['address_family'] == address_family]
    if len(iface_lines) < 1:
        module.fail_json(msg='Error: interface %s not found' % iface)
        return (changed, None)
    iface_options = get_interface_options(iface_lines)
    target_options = get_target_options(iface_options, option)
    if state == 'present':
        if len(target_options) < 1:
            changed = True
            last_line_dict = iface_lines[-1]
            changed, lines = addOptionAfterLine(option, value, iface, lines, last_line_dict, iface_options, address_family)
        elif option in ['pre-up', 'up', 'down', 'post-up']:
            if len(list(filter(lambda i: i['value'] == value, target_options))) < 1:
                changed, lines = addOptionAfterLine(option, value, iface, lines, target_options[-1], iface_options, address_family)
        elif target_options[-1]['value'] != value:
            changed = True
            target_option = target_options[-1]
            line = update_existing_option_line(target_option, value)
            address_family = target_option['address_family']
            index = len(lines) - lines[::-1].index(target_option) - 1
            lines[index] = optionDict(line, iface, option, value, address_family)
    elif state == 'absent':
        if len(target_options) >= 1:
            if option in ['pre-up', 'up', 'down', 'post-up'] and value is not None and (value != 'None'):
                for target_option in [ito for ito in target_options if ito['value'] == value]:
                    changed = True
                    lines = [ln for ln in lines if ln != target_option]
            else:
                changed = True
                for target_option in target_options:
                    lines = [ln for ln in lines if ln != target_option]
    else:
        module.fail_json(msg='Error: unsupported state %s, has to be either present or absent' % state)
    return (changed, lines)