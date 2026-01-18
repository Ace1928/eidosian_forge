from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import GcpSession
def gcp_kms_encrypt(plaintext, **kwargs):
    return GcpKmsFilter().run('encrypt', plaintext=plaintext, **kwargs)