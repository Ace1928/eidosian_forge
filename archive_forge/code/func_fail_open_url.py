from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def fail_open_url(self, e, msg, **kwargs):
    try:
        if isinstance(e, HTTPError):
            msg = '%s: %s' % (msg, to_native(e.read()))
    except Exception as ingore:
        pass
    self.module.fail_json(msg, **kwargs)