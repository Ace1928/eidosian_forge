from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.consul import (
def normalize_link_obj(api_obj, module_obj, key):
    api_objs = api_obj.get(key)
    module_objs = module_obj.get(key)
    if api_objs is None or module_objs is None:
        return
    name_to_id = {i['Name']: i['ID'] for i in api_objs}
    id_to_name = {i['ID']: i['Name'] for i in api_objs}
    for obj in module_objs:
        identifier = obj.get('ID')
        name = obj.get('Name)')
        if identifier and (not name) and (identifier in id_to_name):
            obj['Name'] = id_to_name[identifier]
        if not identifier and name and (name in name_to_id):
            obj['ID'] = name_to_id[name]