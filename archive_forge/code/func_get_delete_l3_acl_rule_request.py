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
def get_delete_l3_acl_rule_request(self, acl_type, acl_name, seq_num):
    """Get request to delete the rule with given sequence number
        in the specified L3 ACL
        """
    url = self.l3_acl_rule_path.format(acl_name=acl_name, acl_type=acl_type_to_payload_map[acl_type])
    url += '/acl-entry={0}'.format(seq_num)
    return {'path': url, 'method': DELETE}