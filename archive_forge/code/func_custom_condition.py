import time
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
def custom_condition(condition: Dict, resource: ResourceInstance) -> bool:
    if not resource.status or not resource.status.conditions:
        return False
    matches = [x for x in resource.status.conditions if x.type == condition['type']]
    if not matches:
        return False
    match: ResourceField = matches[0]
    if match.status == 'Unknown':
        if match.status == condition['status']:
            if 'reason' not in condition:
                return True
            if condition['reason']:
                return match.reason == condition['reason']
        return False
    status = True if match.status == 'True' else False
    if status == boolean(condition['status'], strict=False):
        if condition.get('reason'):
            return match.reason == condition['reason']
        return True
    return False