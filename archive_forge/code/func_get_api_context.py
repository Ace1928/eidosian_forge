from __future__ import absolute_import, division, print_function
import os
import re
import logging
import sys
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
def get_api_context(module, api_creds):
    api_context = module.params.get('api_context')
    if api_context and module.params.get('avi_disable_session_cache_as_fact'):
        return api_context
    elif api_context and (not module.params.get('avi_disable_session_cache_as_fact')):
        key = '%s:%s:%s' % (api_creds.controller, api_creds.username, api_creds.port)
        return api_context.get(key)
    else:
        return None