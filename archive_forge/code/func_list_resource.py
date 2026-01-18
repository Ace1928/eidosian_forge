from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
@_exception2fail_json(msg='Failed to list resource: {0}')
def list_resource(self, resource, search=None, params=None):
    """
        Execute the ``index`` action on an resource.

        :param resource: Plural name of the api resource to show
        :type resource: str
        :param search: Search string as accepted by the API to limit the results
        :type search: str, optional
        :param params: Lookup parameters (i.e. parent_id for nested entities)
        :type params: Union[dict,None], optional
        """
    if params is None:
        params = {}
    else:
        params = params.copy()
    if search is not None:
        params['search'] = search
    params['per_page'] = PER_PAGE
    params = self._resource_prepare_params(resource, 'index', params)
    return self._resource_call(resource, 'index', params)['results']