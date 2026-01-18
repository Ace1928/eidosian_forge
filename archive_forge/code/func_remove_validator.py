from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
import json
def remove_validator(client, db, collection):
    cmd_doc = OrderedDict([('collMod', collection), ('validator', {}), ('validationLevel', 'off')])
    client[db].command(cmd_doc)