import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from ray import cloudpickle
from ray._private import ray_option_utils
from ray._private.protobuf_compat import message_to_dict
from ray._private.pydantic_compat import (
from ray._private.serialization import pickle_dumps
from ray._private.utils import resources_from_ray_options
from ray.serve._private.constants import (
from ray.serve._private.utils import DEFAULT, DeploymentOptionUpdateType
from ray.serve.config import AutoscalingConfig
from ray.serve.generated.serve_pb2 import AutoscalingConfig as AutoscalingConfigProto
from ray.serve.generated.serve_pb2 import DeploymentConfig as DeploymentConfigProto
from ray.serve.generated.serve_pb2 import DeploymentLanguage
from ray.serve.generated.serve_pb2 import EncodingType as EncodingTypeProto
from ray.serve.generated.serve_pb2 import LoggingConfig as LoggingConfigProto
from ray.serve.generated.serve_pb2 import ReplicaConfig as ReplicaConfigProto
from ray.util.placement_group import VALID_PLACEMENT_GROUP_STRATEGIES
@property
def init_kwargs(self) -> Optional[Tuple[Any]]:
    """The init_kwargs for a Python class.

        This property is only meaningful if deployment_def is a Python class.
        Otherwise, it is None.
        """
    if self._init_kwargs is None:
        self._init_kwargs = cloudpickle.loads(self.serialized_init_kwargs)
    return self._init_kwargs