from __future__ import absolute_import, division, print_function
import logging
import math
import re
from decimal import Decimal
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.logging_handler \
import traceback
from ansible.module_utils.basic import missing_required_lib
def get_powerflex_gateway_host_connection(module_params):
    """Establishes connection with PowerFlex storage system"""
    if HAS_POWERFLEX_SDK:
        conn = PowerFlexClient(gateway_address=module_params['hostname'], gateway_port=module_params['port'], verify_certificate=module_params['validate_certs'], username=module_params['username'], password=module_params['password'], timeout=module_params['timeout'])
        conn.initialize()
        return conn