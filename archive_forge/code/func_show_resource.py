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
@_exception2fail_json(msg='Failed to show resource: {0}')
def show_resource(self, resource, resource_id, params=None):
    """
        Execute the ``show`` action on an entity.

        :param resource: Plural name of the api resource to show
        :type resource: str
        :param resource_id: The ID of the entity to show
        :type resource_id: int
        :param params: Lookup parameters (i.e. parent_id for nested entities)
        :type params: Union[dict,None], optional
        """
    if params is None:
        params = {}
    else:
        params = params.copy()
    params['id'] = resource_id
    params = self._resource_prepare_params(resource, 'show', params)
    return self._resource_call(resource, 'show', params)