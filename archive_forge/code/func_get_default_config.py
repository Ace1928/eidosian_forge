from typing import Optional
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.ddpg.ddpg import DDPG, DDPGConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
@classmethod
@override(DDPG)
def get_default_config(cls) -> AlgorithmConfig:
    return ApexDDPGConfig()