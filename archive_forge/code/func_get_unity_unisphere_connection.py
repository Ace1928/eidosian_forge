from __future__ import absolute_import, division, print_function
import logging
from decimal import Decimal
import re
import traceback
import math
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell.logging_handler \
from ansible.module_utils.basic import missing_required_lib
def get_unity_unisphere_connection(module_params, application_type=None):
    """Establishes connection with Unity array using storops SDK"""
    if HAS_UNITY_SDK:
        conn = UnitySystem(host=module_params['unispherehost'], port=module_params['port'], verify=module_params['validate_certs'], username=module_params['username'], password=module_params['password'], application_type=application_type)
        return conn