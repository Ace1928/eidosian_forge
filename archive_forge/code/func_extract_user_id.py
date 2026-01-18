from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.community.general.plugins.module_utils.scaleway import scaleway_argument_spec, Scaleway
def extract_user_id(raw_organization_dict):
    return raw_organization_dict['organizations'][0]['users'][0]['id']