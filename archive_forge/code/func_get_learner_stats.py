import collections
import logging
import numpy as np
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import GradInfoDict, LearnerStatsDict, ResultDict
@DeveloperAPI
def get_learner_stats(grad_info: GradInfoDict) -> LearnerStatsDict:
    """Return optimization stats reported from the policy.

    .. testcode::
        :skipif: True

        grad_info = worker.learn_on_batch(samples)

        # {"td_error": [...], "learner_stats": {"vf_loss": ..., ...}}

        print(get_stats(grad_info))

    .. testoutput::

        {"vf_loss": ..., "policy_loss": ...}
    """
    if LEARNER_STATS_KEY in grad_info:
        return grad_info[LEARNER_STATS_KEY]
    multiagent_stats = {}
    for k, v in grad_info.items():
        if type(v) is dict:
            if LEARNER_STATS_KEY in v:
                multiagent_stats[k] = v[LEARNER_STATS_KEY]
    return multiagent_stats