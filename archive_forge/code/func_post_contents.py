from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils._text import to_text
def post_contents(self, url, file_contents=None, headers=None, **kwargs):
    """
        This method should be avoided in favor of full_post
        """
    kwargs.update({'data': file_contents, 'headers': headers})
    return self.full_post(url, **kwargs)