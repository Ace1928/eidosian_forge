from __future__ import (absolute_import, division, print_function)
import re
import time
import traceback
from ansible.module_utils.basic import (AnsibleModule, env_fallback,
from ansible.module_utils.common.text.converters import to_text
def session_method_wrapper(f):

    def _wrap(self, url, *args, **kwargs):
        try:
            url = self.endpoint + url
            r = f(self, url, *args, **kwargs)
        except Exception as ex:
            raise HwcClientException(0, 'Sending request failed, error=%s' % ex)
        result = None
        if r.content:
            try:
                result = r.json()
            except Exception as ex:
                raise HwcClientException(0, 'Parsing response to json failed, error: %s' % ex)
        code = r.status_code
        if code not in [200, 201, 202, 203, 204, 205, 206, 207, 208, 226]:
            msg = ''
            for i in ['message', 'error.message']:
                try:
                    msg = navigate_value(result, i)
                    break
                except Exception:
                    pass
            else:
                msg = str(result)
            if code == 404:
                raise HwcClientException404(msg)
            raise HwcClientException(code, msg)
        return result
    return _wrap