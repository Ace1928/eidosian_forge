from __future__ import absolute_import, division, print_function
import json
import re
from ansible_collections.community.rabbitmq.plugins.module_utils.version import LooseVersion as Version
from ansible.module_utils.basic import AnsibleModule
def should_be_deleted(self):
    return any((self._policy_check_by_name(policy) for policy in self._list_policies()))