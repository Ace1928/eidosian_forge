from __future__ import absolute_import, division, print_function
import abc
import binascii
import os
from base64 import b64encode
from datetime import datetime
from hashlib import sha256
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import convert_relative_to_datetime
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
def parse_option_list(option_list):
    critical_options = []
    directives = []
    extensions = []
    for option in option_list:
        if option.lower() in _DIRECTIVES:
            directives.append(option.lower())
        else:
            option_object = OpensshCertificateOption.from_string(option)
            if option_object.type == 'critical':
                critical_options.append(option_object)
            else:
                extensions.append(option_object)
    return (critical_options, list(set(extensions + apply_directives(directives))))