from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.consul import (
def list_sessions_for_node(consul_module, node, datacenter):
    return consul_module.get(('session', 'node', node), params={'dc': datacenter})