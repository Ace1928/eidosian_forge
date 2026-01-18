from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def get_number_and_section_from_position(payload, connection, version, api_call_object):
    show_rulebase_command = get_relevant_show_rulebase_command(api_call_object)
    if 'position' in payload:
        section_name = None
        if type(payload['position']) is not dict:
            position = payload['position']
            if position == 'top':
                position = 1
                return (position, section_name)
            elif position == 'bottom':
                show_rulebase_payload = get_relevant_show_rulebase_identifier_payload(api_call_object, payload)
                position = get_rules_amount(connection, version, show_rulebase_payload, show_rulebase_command)
                show_rulebase_payload.update({'offset': position - 1})
                code, response = send_request(connection, version, show_rulebase_command, show_rulebase_payload)
                rulebase = reversed(response['rulebase'])
            else:
                position = int(position)
                payload_for_show_obj_rulebase = build_rulebase_payload(api_call_object, payload, position)
                code, response = send_request(connection, version, show_rulebase_command, payload_for_show_obj_rulebase)
                rulebase = response['rulebase']
                if position > response['total']:
                    raise ValueError('The given position ' + str(position) + ' of rule ' + payload['name'] + 'exceeds the total amount of rules in the rulebase')
                i = 0
                for rules in rulebase:
                    if 'rulebase' in rules and len(rules['rulebase']) == 0:
                        i += 1
                rulebase = rulebase[i:]
            for rules in rulebase:
                if 'rulebase' in rules:
                    section_name = rules['name']
                    return (position, section_name)
                else:
                    return (position, section_name)
        else:
            search_entire_rulebase = payload['search-entire-rulebase']
            position = None
            above_relative_position = False
            pos_before_relative_empty_section = 1
            show_rulebase_payload = get_relevant_show_rulebase_identifier_payload(api_call_object, payload)
            if not search_entire_rulebase:
                code, response = send_request(connection, version, show_rulebase_command, show_rulebase_payload)
                rulebase = response['rulebase']
                position, section_name, above_relative_position, pos_before_relative_empty_section = get_number_and_section_from_relative_position(payload, connection, version, rulebase, above_relative_position, pos_before_relative_empty_section, api_call_object)
            else:
                layer_or_package_payload = get_relevant_layer_or_package_identifier(api_call_object, payload)
                rules_amount = get_rules_amount(connection, version, show_rulebase_payload, show_rulebase_command)
                relative_pos_is_section = relative_position_is_section(connection, version, api_call_object, layer_or_package_payload, payload['position'])
                rulebase_generator = get_rulebase_generator(connection, version, show_rulebase_payload, show_rulebase_command, rules_amount)
                prev_section = None
                for rulebase in rulebase_generator:
                    position, section_name, above_relative_position, pos_before_relative_empty_section, prev_section = get_number_and_section_from_relative_position(payload, connection, version, rulebase, above_relative_position, pos_before_relative_empty_section, api_call_object, prev_section, section_name)
                    if not keep_searching_rulebase(position, section_name, payload['position'], relative_pos_is_section):
                        break
            return (position, section_name)
    return (None, None)