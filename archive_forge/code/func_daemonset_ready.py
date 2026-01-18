import time
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
def daemonset_ready(daemonset: ResourceInstance) -> bool:
    return bool(daemonset.status and daemonset.status.desiredNumberScheduled is not None and (daemonset.status.updatedNumberScheduled == daemonset.status.desiredNumberScheduled) and (daemonset.status.numberReady == daemonset.status.desiredNumberScheduled) and (daemonset.status.observedGeneration == daemonset.metadata.generation) and (not daemonset.status.unavailableReplicas))