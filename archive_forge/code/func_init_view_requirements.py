from gymnasium.spaces import Box
import numpy as np
import random
import tree  # pip install dm_tree
from typing import (
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights, TensorStructType, TensorType
@override(Policy)
def init_view_requirements(self):
    super().init_view_requirements()
    vr = self.view_requirements[SampleBatch.INFOS]
    vr.used_for_training = False
    vr.used_for_compute_actions = False