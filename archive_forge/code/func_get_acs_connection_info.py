from __future__ import (absolute_import, division, print_function)
import os
import json
import traceback
from ansible.module_utils.basic import env_fallback
def get_acs_connection_info(params):
    ecs_params = dict(acs_access_key_id=params.get('alicloud_access_key'), acs_secret_access_key=params.get('alicloud_secret_key'), security_token=params.get('alicloud_security_token'), ecs_role_name=params.get('ecs_role_name'), user_agent='Ansible-Provider-Alicloud')
    return ecs_params