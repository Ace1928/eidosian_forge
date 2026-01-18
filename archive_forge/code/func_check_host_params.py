from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves.urllib import parse as urllib_parse
from mimetypes import MimeTypes
import os
import json
import traceback
def check_host_params(self):
    if self.params['url'] is not None and any((self.params[k] is not None for k in ['proto', 'host', 'port', 'password', 'username', 'vhost'])):
        self.module.fail_json(msg='url and proto, host, port, vhost, username or password cannot be specified at the same time.')
    if self.params['url'] is None and any((self.params[k] is None for k in ['proto', 'host', 'port', 'password', 'username', 'vhost'])):
        self.module.fail_json(msg='Connection parameters must be passed via url, or,  proto, host, port, vhost, username or password.')