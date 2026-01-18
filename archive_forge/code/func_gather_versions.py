import traceback
from typing import Optional
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_text
def gather_versions() -> dict:
    versions = {}
    try:
        import jsonpatch
        versions['jsonpatch'] = jsonpatch.__version__
    except ImportError:
        pass
    try:
        import kubernetes
        versions['kubernetes'] = kubernetes.__version__
    except ImportError:
        pass
    try:
        import kubernetes_validate
        versions['kubernetes-validate'] = kubernetes_validate.__version__
    except ImportError:
        pass
    try:
        import yaml
        versions['pyyaml'] = yaml.__version__
    except ImportError:
        pass
    return versions