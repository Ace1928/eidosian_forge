from __future__ import absolute_import, division, print_function
import random
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url
def query_by_id(self, resource_id=None, path=None, result_key=None, skip_transform=True):
    path = path or self.resource_path
    result_key = result_key or self.ressource_result_key_singular
    resource = self.api_query(path='%s%s' % (path, '/' + resource_id if resource_id else resource_id))
    if resource:
        if skip_transform:
            return resource[result_key]
        else:
            return self.transform_resource(resource[result_key])
    return dict()