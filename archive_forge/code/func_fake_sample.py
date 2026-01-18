import logging
import psutil
from typing import Optional, Any
import numpy as np
from ray.rllib.utils import deprecation_warning
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.replay_buffers import (
from ray.rllib.policy.sample_batch import concat_samples, MultiAgentBatch, SampleBatch
from ray.rllib.utils.typing import ResultDict, SampleBatchType, AlgorithmConfigDict
from ray.util import log_once
def fake_sample(_: Any=None, **kwargs) -> Optional[SampleBatchType]:
    """Always returns a predefined batch.

        Args:
            _: dummy arg to match signature of sample() method
            __: dummy arg to match signature of sample() method
            ``**kwargs``: dummy args to match signature of sample() method

        Returns:
            Predefined MultiAgentBatch fake_sample_output
        """
    return fake_sample_output