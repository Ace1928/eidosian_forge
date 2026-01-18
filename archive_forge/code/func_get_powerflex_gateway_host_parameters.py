from __future__ import absolute_import, division, print_function
import logging
import math
import re
from decimal import Decimal
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.logging_handler \
import traceback
from ansible.module_utils.basic import missing_required_lib
def get_powerflex_gateway_host_parameters():
    """Provides common access parameters required for the
    ansible modules on PowerFlex Storage System"""
    return dict(hostname=dict(type='str', aliases=['gateway_host'], required=True), username=dict(type='str', required=True), password=dict(type='str', required=True, no_log=True), validate_certs=dict(type='bool', aliases=['verifycert'], required=False, default=True), port=dict(type='int', required=False, default=443), timeout=dict(type='int', required=False, default=120))