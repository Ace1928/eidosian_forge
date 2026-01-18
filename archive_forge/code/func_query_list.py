from __future__ import absolute_import, division, print_function
import random
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url
def query_list(self, path=None, result_key=None, query_params=None):
    path = path or self.resource_path
    result_key = result_key or self.ressource_result_key_plural
    resources = self.api_query(path=path, query_params=query_params)
    return resources[result_key] if resources else []