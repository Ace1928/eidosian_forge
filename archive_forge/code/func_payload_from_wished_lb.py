from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_REGIONS, SCALEWAY_ENDPOINT, scaleway_argument_spec, Scaleway
def payload_from_wished_lb(wished_lb):
    return {'organization_id': wished_lb['organization_id'], 'name': wished_lb['name'], 'tags': wished_lb['tags'], 'description': wished_lb['description']}