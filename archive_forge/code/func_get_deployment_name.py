from typing import Any, Dict, List, Optional, Tuple
from ray.dag import DAGNode
from ray.dag.format_utils import get_dag_node_str
from ray.serve._private.constants import RAY_SERVE_ENABLE_NEW_HANDLE_API
from ray.serve.deployment import Deployment
from ray.serve.handle import DeploymentHandle, RayServeHandle
def get_deployment_name(self):
    return self._deployment.name