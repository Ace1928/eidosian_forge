from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.api import WapiModule
from ..module_utils.api import NIOS_DTC_TOPOLOGY
from ..module_utils.api import normalize_ib_spec
def rules_transform(module):
    rule_list = list()
    dest_obj = None
    if not module.params['rules']:
        return rule_list
    for rule in module.params['rules']:
        if rule['dest_type'] == 'POOL':
            dest_obj = wapi.get_object('dtc:pool', {'name': rule['destination_link']})
        else:
            dest_obj = wapi.get_object('dtc:server', {'name': rule['destination_link']})
        if not dest_obj and rule['return_type'] == 'REGULAR':
            module.fail_json(msg='destination_link %s does not exist' % rule['destination_link'])
        tf_rule = dict(dest_type=rule['dest_type'], destination_link=dest_obj[0]['_ref'] if dest_obj else None, return_type=rule['return_type'])
        if rule['sources']:
            tf_rule['sources'] = sources_transform(rule['sources'], module)
        rule_list.append(tf_rule)
    return rule_list