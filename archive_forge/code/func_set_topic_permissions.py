from __future__ import absolute_import, division, print_function
import ansible_collections.community.rabbitmq.plugins.module_utils.version as Version
import json
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.collections import count
def set_topic_permissions(self):
    permissions_to_add = list()
    for vhost_exchange, permission_dict in self.topic_permissions.items():
        if permission_dict != self.existing_topic_permissions.get(vhost_exchange, {}):
            permissions_to_add.append(permission_dict)
    permissions_to_clear = list()
    for vhost_exchange in self.existing_topic_permissions.keys():
        if vhost_exchange not in self.topic_permissions:
            permissions_to_clear.append(vhost_exchange)
    for vhost_exchange in permissions_to_clear:
        vhost, exchange = vhost_exchange
        cmd = 'clear_topic_permissions -p {vhost} {username} {exchange}'.format(username=self.username, vhost=vhost, exchange=exchange)
        self._exec(cmd.split(' '))
    for permissions in permissions_to_add:
        cmd = 'set_topic_permissions -p {vhost} {username} {exchange} {write} {read}'.format(username=self.username, **permissions)
        self._exec(cmd.split(' '))
    self.existing_topic_permissions = self._get_topic_permissions()