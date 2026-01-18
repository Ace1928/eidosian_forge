from __future__ import absolute_import, division, print_function
from ansible_collections.community.crypto.plugins.module_utils.ecs.api import (
import datetime
import os
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
def ecs_certificate_argument_spec():
    return dict(backup=dict(type='bool', default=False), force=dict(type='bool', default=False), path=dict(type='path', required=True), full_chain_path=dict(type='path'), tracking_id=dict(type='int'), remaining_days=dict(type='int', default=30), request_type=dict(type='str', default='new', choices=['new', 'renew', 'reissue', 'validate_only']), cert_type=dict(type='str', choices=['STANDARD_SSL', 'ADVANTAGE_SSL', 'UC_SSL', 'EV_SSL', 'WILDCARD_SSL', 'PRIVATE_SSL', 'PD_SSL', 'CODE_SIGNING', 'EV_CODE_SIGNING', 'CDS_INDIVIDUAL', 'CDS_GROUP', 'CDS_ENT_LITE', 'CDS_ENT_PRO', 'SMIME_ENT']), csr=dict(type='str'), subject_alt_name=dict(type='list', elements='str'), eku=dict(type='str', choices=['SERVER_AUTH', 'CLIENT_AUTH', 'SERVER_AND_CLIENT_AUTH']), ct_log=dict(type='bool'), client_id=dict(type='int', default=1), org=dict(type='str'), ou=dict(type='list', elements='str'), end_user_key_storage_agreement=dict(type='bool'), tracking_info=dict(type='str'), requester_name=dict(type='str', required=True), requester_email=dict(type='str', required=True), requester_phone=dict(type='str', required=True), additional_emails=dict(type='list', elements='str'), custom_fields=dict(type='dict', default=None, options=custom_fields_spec()), cert_expiry=dict(type='str'), cert_lifetime=dict(type='str', choices=['P1Y', 'P2Y', 'P3Y']))