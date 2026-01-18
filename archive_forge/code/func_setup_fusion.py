from __future__ import absolute_import, division, print_function
from ansible_collections.purestorage.fusion.plugins.module_utils.errors import (
from ansible_collections.purestorage.fusion.plugins.module_utils.prerequisites import (
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
def setup_fusion(module):
    check_dependencies(module)
    install_fusion_exception_hook(module)
    return get_fusion(module)