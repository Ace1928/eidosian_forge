from __future__ import absolute_import, division, print_function
import copy
import json
from ansible_collections.kubernetes.core.plugins.module_utils.ansiblemodule import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.service import (
def get_exception_message(exc):
    try:
        d = json.loads(exc.body.decode('utf8'))
        return d['message']
    except Exception:
        return exc