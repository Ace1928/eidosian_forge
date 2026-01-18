from __future__ import absolute_import, division, print_function
import os
from traceback import format_exc
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.urls import fetch_url
def add_http_proxy(cmd):
    for envvar in ('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy'):
        proxy = os.environ.get(envvar)
        if proxy:
            break
    if proxy:
        cmd += ' --keyserver-options http-proxy=%s' % proxy
    return cmd