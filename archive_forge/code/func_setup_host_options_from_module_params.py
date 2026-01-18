from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def setup_host_options_from_module_params(host_options, module, keys):
    """if an option is not set, use primary value.
       but don't mix up basic and certificate authentication methods

       host_options is updated in place
       option values are read from module.params
       keys is a list of keys that need to be added/updated/left alone in host_options
    """
    password_keys = ['username', 'password']
    certificate_keys = ['cert_filepath', 'key_filepath']
    use_password = any((host_options.get(x) is not None for x in password_keys))
    use_certificate = any((host_options.get(x) is not None for x in certificate_keys))
    if use_password and use_certificate:
        module.fail_json(msg='Error: host cannot have both basic authentication (username/password) and certificate authentication (cert/key files).')
    if use_password:
        exclude_keys = certificate_keys
    elif use_certificate:
        exclude_keys = password_keys
    else:
        exclude_keys = []
    for key in keys:
        if host_options.get(key) is None and key not in exclude_keys:
            host_options[key] = module.params[key]