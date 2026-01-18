from __future__ import (absolute_import, division, print_function)
import traceback
from datetime import datetime
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.okd.plugins.module_utils.openshift_ldap import (
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def create_openshift_groups(self, groups: list):
    diffs = []
    results = []
    changed = False
    for definition in groups:
        name = definition['metadata']['name']
        existing = self.get_group_info(name=name)
        if not self.module.check_mode:
            method = 'patch' if existing else 'create'
            try:
                if existing:
                    definition = self.k8s_group_api.patch(definition).to_dict()
                else:
                    definition = self.k8s_group_api.create(definition).to_dict()
            except DynamicApiError as exc:
                self.module.fail_json(msg="Failed to %s Group '%s' due to: %s" % (method, name, exc.body))
            except Exception as exc:
                self.module.fail_json(msg="Failed to %s Group '%s' due to: %s" % (method, name, to_native(exc)))
        equals = False
        if existing:
            equals, diff = self.module.diff_objects(existing, definition)
            diffs.append(diff)
        changed = changed or not equals
        results.append(definition)
    return (results, diffs, changed)