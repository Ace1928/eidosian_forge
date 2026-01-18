from __future__ import absolute_import, division, print_function
import json
import os
import random
import mimetypes
from pprint import pformat
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils._text import to_native
import ssl
def is_rest(self, used_unsupported_rest_properties=None):
    """ only return error if there is a reason to """
    use_rest, error = self._is_rest(used_unsupported_rest_properties)
    if used_unsupported_rest_properties is None:
        return use_rest
    return (use_rest, error)