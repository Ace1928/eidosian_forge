import logging
import json
import os
from packaging import version
import re
from typing import Any, Dict, Union
import ray
from ray.rllib.utils.serialization import NOT_SERIALIZABLE, serialize_type
from ray.train import Checkpoint
from ray.util import log_once
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
def convert_to_msgpack_policy_checkpoint(policy_checkpoint: Union[str, Checkpoint], msgpack_checkpoint_dir: str) -> str:
    """Converts a Policy checkpoint (pickle based) to a msgpack based one.

    Msgpack has the advantage of being python version independent.

    Args:
        policy_checkpoint: The directory, in which to find the Policy checkpoint (pickle
            based).
        msgpack_checkpoint_dir: The directory, in which to create the new msgpack
            based checkpoint.

    Returns:
        The directory in which the msgpack checkpoint has been created. Note that
        this is the same as `msgpack_checkpoint_dir`.
    """
    from ray.rllib.policy.policy import Policy
    policy = Policy.from_checkpoint(policy_checkpoint)
    os.makedirs(msgpack_checkpoint_dir, exist_ok=True)
    policy.export_checkpoint(msgpack_checkpoint_dir, policy_state=policy.get_state(), checkpoint_format='msgpack')
    del policy
    return msgpack_checkpoint_dir