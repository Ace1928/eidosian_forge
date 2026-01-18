from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_export_policy_id(rest_api, policy_name, svm_name, module):
    api = 'protocols/nfs/export-policies'
    query = {'name': policy_name, 'svm.name': svm_name}
    record, error = rest_generic.get_one_record(rest_api, api, query)
    if error:
        module.fail_json(msg='Could not find export policy %s on SVM %s' % (policy_name, svm_name))
    return record['id'] if record else None