from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def destination_epg_spec():
    return dict(tenant=dict(type='str', required=True, aliases=['tenant_name']), ap=dict(type='str', required=True, aliases=['ap_name', 'app_profile', 'app_profile_name']), epg=dict(type='str', required=True, aliases=['epg_name']), source_ip=dict(type='str', required=True), destination_ip=dict(type='str', required=True), span_version=dict(type='str', choices=['version_1', 'version_2']), version_enforced=dict(type='bool'), flow_id=dict(type='int'), ttl=dict(type='int'), mtu=dict(type='int'), dscp=dict(type='str', choices=['CS0', 'CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6', 'CS7', 'EF', 'VA', 'AF11', 'AF12', 'AF13', 'AF21', 'AF22', 'AF23', 'AF31', 'AF32', 'AF33', 'AF41', 'AF42', 'AF43', 'unspecified']))