import traceback
from typing import Optional
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_text
def has_at_least(self, dependency: str, minimum: Optional[str]=None, warn: bool=False) -> bool:
    supported = has_at_least(dependency, minimum)
    if not supported and warn:
        self.warn('{0}<{1} is not supported or tested. Some features may not work.'.format(dependency, minimum))
    return supported