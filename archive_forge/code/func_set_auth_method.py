from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def set_auth_method(module, username, password, cert_filepath, key_filepath):
    error = None
    if password is None and username is None:
        if cert_filepath is None:
            error = 'Error: cannot have a key file without a cert file' if key_filepath is not None else 'Error: ONTAP module requires username/password or SSL certificate file(s)'
        else:
            auth_method = 'single_cert' if key_filepath is None else 'cert_key'
    elif password is not None and username is not None:
        if cert_filepath is not None or key_filepath is not None:
            error = 'Error: cannot have both basic authentication (username/password) ' + 'and certificate authentication (cert/key files)'
        else:
            auth_method = 'basic_auth' if has_feature(module, 'classic_basic_authorization') else 'speedy_basic_auth'
    else:
        error = 'Error: username and password have to be provided together'
        if cert_filepath is not None or key_filepath is not None:
            error += ' and cannot be used with cert or key files'
    if error:
        module.fail_json(msg=error)
    return auth_method