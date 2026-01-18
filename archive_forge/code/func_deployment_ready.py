import time
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
def deployment_ready(deployment: ResourceInstance) -> bool:
    return bool(deployment.status and deployment.spec.replicas == (deployment.status.replicas or 0) and (deployment.status.availableReplicas == deployment.status.replicas) and (deployment.status.observedGeneration == deployment.metadata.generation) and (not deployment.status.unavailableReplicas))