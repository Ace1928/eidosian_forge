from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.scaleway import (
from ansible.module_utils.basic import AnsibleModule
def payload_from_wished_cr(wished_cr):
    payload = {'project_id': wished_cr['project_id'], 'name': wished_cr['name'], 'description': wished_cr['description'], 'is_public': wished_cr['privacy_policy'] == 'public'}
    return payload