from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def get_operation_docs(op):
    op_url = op[OperationField.URL][len(self._base_path):]
    return docs[PropName.PATHS].get(op_url, {}).get(op[OperationField.METHOD], {})