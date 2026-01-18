from __future__ import (absolute_import, division, print_function)
import traceback
from datetime import datetime
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.okd.plugins.module_utils.openshift_ldap import (
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def delete_openshift_group(self, name: str):
    result = dict(kind=self.kind, apiVersion=self.version, metadata=dict(name=name))
    if not self.module.check_mode:
        try:
            result = self.k8s_group_api.delete(name=name).to_dict()
        except DynamicApiError as exc:
            self.module.fail_json(msg="Failed to delete Group '{0}' due to: {1}".format(name, exc.body))
        except Exception as exc:
            self.module.fail_json(msg="Failed to delete Group '{0}' due to: {1}".format(name, to_native(exc)))
    return result