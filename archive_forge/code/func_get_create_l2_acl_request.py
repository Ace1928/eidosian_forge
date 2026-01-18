from __future__ import absolute_import, division, print_function
from ast import literal_eval
from ansible.module_utils._text import to_text
from ansible.module_utils.common.validation import check_required_arguments
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_create_l2_acl_request(self, acl_name):
    """Get request to create L2 ACL with specified name"""
    url = self.acl_path
    payload = {'acl-set': [{'name': acl_name, 'type': L2_ACL_TYPE, 'config': {'name': acl_name, 'type': L2_ACL_TYPE}}]}
    return {'path': url, 'method': PATCH, 'data': payload}