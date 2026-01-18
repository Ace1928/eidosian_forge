from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.quoting import (
from ansible_collections.community.routeros.plugins.module_utils.api import (
import re
def api_arbitrary(self):
    param = {}
    self.arbitrary = self.split_params(self.arbitrary)
    arb_cmd = self.arbitrary[0]
    if len(self.arbitrary) > 1:
        param = self.list_to_dic(self.arbitrary[1:])
    try:
        arbitrary_result = self.api_path(arb_cmd, **param)
        for i in arbitrary_result:
            self.result['message'].append(i)
        self.return_result(False)
    except LibRouterosError as e:
        self.errors(e)