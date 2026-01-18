import collections
import logging
import numpy as np
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import GradInfoDict, LearnerStatsDict, ResultDict
@DeveloperAPI
def collect_metrics(workers: 'WorkerSet', remote_worker_ids: Optional[List[int]]=None, timeout_seconds: int=180, keep_custom_metrics: bool=False) -> ResultDict:
    """Gathers episode metrics from rollout worker set.

    Args:
        workers: WorkerSet.
        remote_worker_ids: Optional list of IDs of remote workers to collect
            metrics from.
        timeout_seconds: Timeout in seconds for collecting metrics from remote workers.
        keep_custom_metrics: Whether to keep custom metrics in the result dict as
            they are (True) or to aggregate them (False).

    Returns:
        A result dict of metrics.
    """
    episodes = collect_episodes(workers, remote_worker_ids, timeout_seconds=timeout_seconds)
    metrics = summarize_episodes(episodes, episodes, keep_custom_metrics=keep_custom_metrics)
    return metrics