import json
import logging
from abc import ABC
from copy import deepcopy
from typing import Any, Dict, List, Optional
from zlib import crc32
from ray._private.pydantic_compat import BaseModel
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.utils import DeploymentOptionUpdateType, get_random_letters
from ray.serve.generated.serve_pb2 import DeploymentVersion as DeploymentVersionProto
def compute_hashes(self):
    serialized_ray_actor_options = _serialize(self.ray_actor_options or {})
    self.ray_actor_options_hash = crc32(serialized_ray_actor_options)
    combined_placement_group_options = {}
    if self.placement_group_bundles is not None:
        combined_placement_group_options['bundles'] = self.placement_group_bundles
    if self.placement_group_strategy is not None:
        combined_placement_group_options['strategy'] = self.placement_group_strategy
    serialized_placement_group_options = _serialize(combined_placement_group_options)
    self.placement_group_options_hash = crc32(serialized_placement_group_options)
    self.reconfigure_actor_hash = crc32(self._get_serialized_options([DeploymentOptionUpdateType.NeedsActorReconfigure]))
    self._hash = crc32(self.code_version.encode('utf-8') + serialized_ray_actor_options + serialized_placement_group_options + str(self.max_replicas_per_node).encode('utf-8') + self._get_serialized_options([DeploymentOptionUpdateType.NeedsReconfigure, DeploymentOptionUpdateType.NeedsActorReconfigure]))