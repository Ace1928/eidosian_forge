from __future__ import absolute_import, division, print_function
import os
from random import randrange
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate import (
def generate_serial_number():
    """Generate a serial number for a certificate"""
    while True:
        result = randrange(0, 1 << 160)
        if result >= 1000:
            return result