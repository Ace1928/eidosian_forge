from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def check_pod(module, array):
    """Check is the requested pod to create volume in exists"""
    pod_exists = False
    api_version = array._list_available_rest_versions()
    if POD_API_VERSION in api_version:
        pod_name = module.params['name'].split('::')[0]
        try:
            pods = array.list_pods()
        except Exception:
            module.fail_json(msg='Failed to get pod list. Check array.')
        for pod in range(0, len(pods)):
            if pod_name == pods[pod]['name']:
                pod_exists = True
                break
    else:
        module.fail_json(msg='Pod volumes are not supported. Please upgrade your FlashArray.')
    return pod_exists