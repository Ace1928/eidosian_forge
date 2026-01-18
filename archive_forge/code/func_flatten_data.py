from typing import Any, Callable, Type
import numpy as np
import tree  # dm_tree
from ray.rllib.connectors.connector import (
from ray.rllib.connectors.registry import register_connector
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
def flatten_data(data: AgentConnectorsOutput):
    assert isinstance(data, AgentConnectorsOutput), 'Single agent data must be of type AgentConnectorsOutput'
    raw_dict = data.raw_dict
    sample_batch = data.sample_batch
    flattened = {}
    for k, v in sample_batch.items():
        if k in [SampleBatch.INFOS, SampleBatch.ACTIONS] or k.startswith('state_out_'):
            flattened[k] = v
            continue
        if v is None:
            flattened[k] = None
            continue
        flattened[k] = np.array(tree.flatten(v))
    flattened = SampleBatch(flattened, is_training=False)
    return AgentConnectorsOutput(raw_dict, flattened)