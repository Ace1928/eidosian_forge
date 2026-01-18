from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
def parse_PEM_list(module, text, source, fail_on_error=True):
    """
    Parse concatenated PEM certificates. Return list of ``Certificate`` objects.
    """
    result = []
    for cert_pem in split_pem_list(text):
        try:
            cert = cryptography.x509.load_pem_x509_certificate(to_bytes(cert_pem), _cryptography_backend)
            result.append(Certificate(cert_pem, cert))
        except Exception as e:
            msg = 'Cannot parse certificate #{0} from {1}: {2}'.format(len(result) + 1, source, e)
            if fail_on_error:
                module.fail_json(msg=msg)
            else:
                module.warn(msg)
    return result