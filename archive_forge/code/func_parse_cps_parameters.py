from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import dict_diff
def parse_cps_parameters(module_name, qualifier, attr_type, attr_data, operation=None, db=None, commit_event=None):
    obj = cps_object.CPSObject(module=module_name, qual=qualifier)
    if operation:
        obj.set_property('oper', operation)
    if attr_type:
        for key, val in iteritems(attr_type):
            cps_utils.cps_attr_types_map.add_type(key, val)
    for key, val in iteritems(attr_data):
        embed_attrs = key.split(',')
        embed_attrs_len = len(embed_attrs)
        if embed_attrs_len >= 3:
            obj.add_embed_attr(embed_attrs, val, embed_attrs_len - 2)
        elif isinstance(val, str):
            val_list = val.split(',')
            if len(val_list) == 1 or val.startswith('{'):
                obj.add_attr(key, val)
            else:
                obj.add_attr(key, val_list)
        else:
            obj.add_attr(key, val)
    if db:
        cps.set_ownership_type(obj.get_key(), 'db')
        obj.set_property('db', True)
    else:
        obj.set_property('db', False)
    if commit_event:
        cps.set_auto_commit_event(obj.get_key(), True)
        obj.set_property('commit-event', True)
    return obj