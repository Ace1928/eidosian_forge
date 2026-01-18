from __future__ import absolute_import, division, print_function
import base64
import re
import textwrap
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves.urllib.parse import unquote
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import ModuleFailException
def der_to_pem(der_cert):
    """
    Convert the DER format certificate in der_cert to a PEM format certificate and return it.
    """
    return '-----BEGIN CERTIFICATE-----\n{0}\n-----END CERTIFICATE-----\n'.format('\n'.join(textwrap.wrap(base64.b64encode(der_cert).decode('utf8'), 64)))