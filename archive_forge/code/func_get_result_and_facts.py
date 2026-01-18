from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def get_result_and_facts(self, facts_name, resource):
    result = self.get_result(resource)
    ansible_facts = {facts_name: result.copy()}
    for k in ['diff', 'changed']:
        if k in ansible_facts[facts_name]:
            del ansible_facts[facts_name][k]
    result.update(ansible_facts=ansible_facts)
    return result