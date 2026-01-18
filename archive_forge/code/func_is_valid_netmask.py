from __future__ import absolute_import, division, print_function
import logging
from decimal import Decimal
import re
import traceback
import math
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell.logging_handler \
from ansible.module_utils.basic import missing_required_lib
def is_valid_netmask(netmask):
    """Validates if ip is valid subnet mask"""
    if netmask:
        regexp = re.compile('^((128|192|224|240|248|252|254)\\.0\\.0\\.0)|(255\\.(((0|128|192|224|240|248|252|254)\\.0\\.0)|(255\\.(((0|128|192|224|240|248|252|254)\\.0)|255\\.(0|128|192|224|240|248|252|254)))))$')
        if not regexp.search(netmask):
            return False
        return True