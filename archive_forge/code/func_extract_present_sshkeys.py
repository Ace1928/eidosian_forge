from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.community.general.plugins.module_utils.scaleway import scaleway_argument_spec, Scaleway
def extract_present_sshkeys(raw_organization_dict):
    ssh_key_list = raw_organization_dict['organizations'][0]['users'][0]['ssh_public_keys']
    ssh_key_lookup = [ssh_key['key'] for ssh_key in ssh_key_list]
    return ssh_key_lookup