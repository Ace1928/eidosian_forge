from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native, to_bytes
from hashlib import sha1
def encrypt_cleartext_password(password_to_encrypt, encryption_method):
    encrypted_password = encryption_method(password_to_encrypt)
    return encrypted_password