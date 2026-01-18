import time
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
def statefulset_ready(statefulset: ResourceInstance) -> bool:
    updated_replicas = statefulset.status.updatedReplicas or 0
    ready_replicas = statefulset.status.readyReplicas or 0
    return bool(statefulset.status and statefulset.spec.updateStrategy.type == 'RollingUpdate' and (statefulset.status.observedGeneration == (statefulset.metadata.generation or 0)) and (statefulset.status.updateRevision == statefulset.status.currentRevision) and (updated_replicas == statefulset.spec.replicas) and (ready_replicas == statefulset.spec.replicas) and (statefulset.status.replicas == statefulset.spec.replicas))