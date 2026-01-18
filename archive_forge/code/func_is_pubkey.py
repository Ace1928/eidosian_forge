from __future__ import absolute_import, division, print_function
import re
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
def is_pubkey(string):
    """Verifies if string is a pubkey"""
    pgp_regex = '.*?(-----BEGIN PGP PUBLIC KEY BLOCK-----.*?-----END PGP PUBLIC KEY BLOCK-----).*'
    return bool(re.match(pgp_regex, to_native(string, errors='surrogate_or_strict'), re.DOTALL))