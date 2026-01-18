from __future__ import absolute_import, division, print_function
import ansible_collections.community.rabbitmq.plugins.module_utils.version as Version
import json
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.collections import count
def as_topic_permission_dict(topic_permission_list):
    return dict([((perm['vhost'], perm['exchange']), perm) for perm in topic_permission_list])