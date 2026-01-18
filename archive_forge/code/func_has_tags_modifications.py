from __future__ import absolute_import, division, print_function
import ansible_collections.community.rabbitmq.plugins.module_utils.version as Version
import json
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.collections import count
def has_tags_modifications(self):
    return set(self.tags) != set(self.existing_tags)