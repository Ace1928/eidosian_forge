import functools
from collections import defaultdict
import numpy as np
import uuid
import gymnasium as gym
from gymnasium.core import ActType, ObsType
from typing import Any, Dict, List, Optional, SupportsFloat, Union
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.utils import BufferWithInfiniteLookback
def get_sample_batch(self) -> SampleBatch:
    """Converts this `SingleAgentEpisode` into a `SampleBatch`.

        Returns:
            A SampleBatch containing all of this episode's data.
        """
    return SampleBatch(self.get_data_dict())