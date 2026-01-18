from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.community.general.plugins.module_utils.scaleway import scaleway_argument_spec, Scaleway
def sshkey_user_patch(ssh_lookup):
    ssh_list = {'ssh_public_keys': [{'key': key} for key in ssh_lookup]}
    return ssh_list