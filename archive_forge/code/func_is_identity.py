from typing import Any
from ray.rllib.connectors.connector import (
from ray.rllib.connectors.registry import register_connector
from ray.rllib.models.preprocessors import get_preprocessor, NoPreprocessor
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentConnectorDataType
from ray.util.annotations import PublicAPI
def is_identity(self):
    """Returns whether this preprocessor connector is a no-op preprocessor."""
    return isinstance(self._preprocessor, NoPreprocessor)