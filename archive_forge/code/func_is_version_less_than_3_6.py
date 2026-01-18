from __future__ import absolute_import, division, print_function
import logging
import math
import re
from decimal import Decimal
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.logging_handler \
import traceback
from ansible.module_utils.basic import missing_required_lib
def is_version_less_than_3_6(version):
    """Verifies if powerflex version is less than 3.6"""
    version = re.search('R\\s*([\\d.]+)', version.replace('_', '.')).group(1)
    return pkg_resources.parse_version(version) < pkg_resources.parse_version('3.6')