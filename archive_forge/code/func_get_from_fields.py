from __future__ import (absolute_import, division, print_function)
import re
import operator
from functools import reduce
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible.module_utils._text import to_native
def get_from_fields(d, fields):
    try:
        return reduce(operator.getitem, fields, d)
    except Exception:
        return None