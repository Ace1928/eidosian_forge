from collections import OrderedDict
import contextlib
import gymnasium as gym
from gymnasium.spaces import Space
import numpy as np
from typing import Dict, List, Any, Union
from ray.rllib.models.preprocessors import get_preprocessor, RepeatedValuesPreprocessor
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import NullContextManager
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.framework import try_import_tf, try_import_torch, TensorType
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.typing import ModelConfigDict, ModelInputDict, TensorStructType
def import_from_h5(self, h5_file: str) -> None:
    """Imports weights from an h5 file.

        Args:
            h5_file: The h5 file name to import weights from.

        .. testcode::
            :skipif: True

            from ray.rllib.algorithms.ppo import PPO
            algo = PPO(...)
            algo.import_policy_model_from_h5("/tmp/weights.h5")
            for _ in range(10):
                algo.train()
        """
    raise NotImplementedError