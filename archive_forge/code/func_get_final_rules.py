import itertools
import json
import re
from collections import namedtuple
from copy import deepcopy
from ipaddress import ip_network
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.network import to_ipv6_subnet
from ansible.module_utils.common.network import to_subnet
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_id
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def get_final_rules(client, module, security_group_rules, specified_rules, purge_rules):
    if specified_rules is None:
        return security_group_rules
    if purge_rules:
        final_rules = []
    else:
        final_rules = list(security_group_rules)
    specified_rules = flatten_nested_targets(module, deepcopy(specified_rules))
    for rule in specified_rules:
        format_rule = {'from_port': None, 'to_port': None, 'ip_protocol': rule.get('proto'), 'ip_ranges': [], 'ipv6_ranges': [], 'prefix_list_ids': [], 'user_id_group_pairs': []}
        if rule.get('proto') in ('all', '-1', -1):
            format_rule['ip_protocol'] = '-1'
            format_rule.pop('from_port')
            format_rule.pop('to_port')
        elif rule.get('ports'):
            if rule.get('ports') and (isinstance(rule['ports'], string_types) or isinstance(rule['ports'], int)):
                rule['ports'] = [rule['ports']]
            for port in rule.get('ports'):
                if isinstance(port, string_types) and '-' in port:
                    format_rule['from_port'], format_rule['to_port'] = port.split('-')
                else:
                    format_rule['from_port'] = format_rule['to_port'] = port
        elif rule.get('from_port') or rule.get('to_port'):
            format_rule['from_port'] = rule.get('from_port', rule.get('to_port'))
            format_rule['to_port'] = rule.get('to_port', rule.get('from_port'))
        for source_type in ('cidr_ip', 'cidr_ipv6', 'prefix_list_id'):
            if rule.get(source_type):
                rule_key = {'cidr_ip': 'ip_ranges', 'cidr_ipv6': 'ipv6_ranges', 'prefix_list_id': 'prefix_list_ids'}.get(source_type)
                if rule.get('rule_desc'):
                    format_rule[rule_key] = [{source_type: rule[source_type], 'description': rule['rule_desc']}]
                else:
                    if not isinstance(rule[source_type], list):
                        rule[source_type] = [rule[source_type]]
                    format_rule[rule_key] = [{source_type: target} for target in rule[source_type]]
        if rule.get('group_id') or rule.get('group_name'):
            rule_sg = group_exists(client, module, module.params['vpc_id'], rule.get('group_id'), rule.get('group_name'))[0]
            if rule_sg is None:
                format_rule['user_id_group_pairs'] = [{'group_id': rule.get('group_id'), 'group_name': rule.get('group_name'), 'peering_status': None, 'user_id': get_account_id(security_group, module), 'vpc_id': module.params['vpc_id'], 'vpc_peering_connection_id': None}]
            else:
                rule_sg = camel_dict_to_snake_dict(rule_sg)
                format_rule['user_id_group_pairs'] = [{'description': rule_sg.get('description', rule_sg.get('group_desc')), 'group_id': rule_sg.get('group_id', rule.get('group_id')), 'group_name': rule_sg.get('group_name', rule.get('group_name')), 'peering_status': rule_sg.get('peering_status'), 'user_id': rule_sg.get('user_id', get_account_id(security_group, module)), 'vpc_id': rule_sg.get('vpc_id', module.params['vpc_id']), 'vpc_peering_connection_id': rule_sg.get('vpc_peering_connection_id')}]
            for k, v in list(format_rule['user_id_group_pairs'][0].items()):
                if v is None:
                    format_rule['user_id_group_pairs'][0].pop(k)
        final_rules.append(format_rule)
    return final_rules