from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (Config, HwcClientException,
import re
def create_resource_opts(module):
    params = dict()
    v = module.params.get('display_name')
    if not is_empty_value(v):
        params['display_name'] = v
    v = module.params.get('name')
    if not is_empty_value(v):
        params['name'] = v
    return params