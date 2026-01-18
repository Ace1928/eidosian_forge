from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def get_login_domain_id(self, domain):
    """Get a domain and return its id"""
    if domain is None:
        return domain
    d = self.get_obj('auth/login-domains', key='domains', name=domain)
    if not d:
        self.fail_json(msg="Login domain '%s' is not a valid domain name." % domain)
    if 'id' not in d:
        self.fail_json(msg="Login domain lookup failed for domain '%s': %s" % (domain, d))
    return d['id']