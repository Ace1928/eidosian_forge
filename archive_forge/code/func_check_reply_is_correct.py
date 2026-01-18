from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.six.moves.urllib import parse as urllib_parse
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.rabbitmq.plugins.module_utils.rabbitmq import rabbitmq_argument_spec
def check_reply_is_correct(self):
    """
        :return:
        """
    if self.api_result.status_code in self.http_check_states:
        return True
    return False