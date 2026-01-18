from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import construct_ansible_facts, FtdServerError, HTTPMethod
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField
def is_upload_operation(op_spec):
    return op_spec[OperationField.METHOD] == HTTPMethod.POST or 'UploadStatus' in op_spec[OperationField.MODEL_NAME]