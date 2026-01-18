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
def find_compute_resource_parts(self, part_name, name, compute_resource, cluster=None, supported_crs=None):
    if supported_crs is None:
        supported_crs = []
    if compute_resource['provider'].lower() not in supported_crs:
        return {'id': name, 'name': name}
    additional_params = {'id': compute_resource['id']}
    if cluster is not None:
        additional_params['cluster_id'] = six.moves.urllib.parse.quote(cluster['_api_identifier'], safe='')
    api_name = 'available_{0}'.format(part_name)
    available_parts = self.resource_action('compute_resources', api_name, params=additional_params, ignore_check_mode=True, record_change=False)['results']
    part = next((part for part in available_parts if str(part['name']) == str(name) or str(part['id']) == str(name) or part.get('full_path') == str(name)), None)
    if part is None:
        err_msg = "Could not find {0} '{1}' on compute resource '{2}'.".format(part_name, name, compute_resource.get('name'))
        self.fail_json(msg=err_msg)
    return part