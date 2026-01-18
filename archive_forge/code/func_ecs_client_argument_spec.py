from __future__ import absolute_import, division, print_function
import json
import os
import re
import traceback
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import Request
def ecs_client_argument_spec():
    return dict(entrust_api_user=dict(type='str', required=True), entrust_api_key=dict(type='str', required=True, no_log=True), entrust_api_client_cert_path=dict(type='path', required=True), entrust_api_client_cert_key_path=dict(type='path', required=True, no_log=True), entrust_api_specification_path=dict(type='path', default='https://cloud.entrust.net/EntrustCloud/documentation/cms-api-2.1.0.yaml'))