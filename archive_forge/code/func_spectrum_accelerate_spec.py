from __future__ import absolute_import, division, print_function
import traceback
from functools import wraps
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import missing_required_lib
def spectrum_accelerate_spec():
    """ Return arguments spec for AnsibleModule """
    return dict(endpoints=dict(required=True), username=dict(required=True), password=dict(no_log=True, required=True))