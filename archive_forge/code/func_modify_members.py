from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def modify_members(module, config, members):
    """
    Modifies the members section of the config document as appropriate.
    @module - Ansible module object
    @config - Replicaset config document from MongoDB
    @members - Members config from module
    """
    try:
        from collections import OrderedDict
    except ImportError as excep:
        try:
            from ordereddict import OrderedDict
        except ImportError as excep:
            module.fail_json(msg='Cannot import OrderedDict class. You can probably install with: pip install ordereddict: %s' % to_native(excep))
    new_member_config = []
    existing_members = []
    max_id = 0
    if all((isinstance(member, str) for member in members)):
        for current_member in config['members']:
            if current_member['host'] in members:
                new_member_config.append(current_member)
                existing_members.append(current_member['host'])
                if current_member['_id'] > max_id:
                    max_id = current_member['_id']
        member_additions = list(set(members) - set(existing_members))
        if len(member_additions) > 0:
            for member in member_additions:
                if ':' not in member:
                    member += ':27017'
                new_member_config.append(OrderedDict([('_id', max_id + 1), ('host', member)]))
                max_id += 1
        config['members'] = new_member_config
    elif all((isinstance(member, dict) for member in members)):
        new_member_config = []
        existing_members = {}
        matched_members = []
        max_id = 0
        for member in config['members']:
            existing_members[member['host']] = member['_id']
            if member['_id'] > max_id:
                max_id = member['_id']
        for member in members:
            if member['host'] in existing_members:
                member['_id'] = existing_members[member['host']]
                matched_members.append(member['host'])
                new_member_config.append(member)
        for member in members:
            if member['host'] not in matched_members:
                max_id = max_id + 1
                member['_id'] = max_id
                new_member_config.append(member)
        config['members'] = new_member_config
    else:
        module.fail_json(msg='All items in members must be either of type dict of str')
    return config