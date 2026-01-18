from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
from . import client
def get_spec_payload(source, *wanted_params):
    return dict(((k, source[k]) for k in wanted_params if source.get(k) is not None))