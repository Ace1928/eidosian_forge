from __future__ import (absolute_import, division, print_function)
import traceback
from datetime import datetime
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.okd.plugins.module_utils.openshift_ldap import (
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def get_group_info(self, return_list=False, **kwargs):
    params = dict(kind=self.kind, api_version=self.version)
    params.update(kwargs)
    result = self.module.kubernetes_facts(**params)
    if len(result['resources']) == 0:
        return None
    if len(result['resources']) == 1 and (not return_list):
        return result['resources'][0]
    else:
        return result['resources']