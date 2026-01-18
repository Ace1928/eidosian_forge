import collections
import logging
import numpy as np
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import GradInfoDict, LearnerStatsDict, ResultDict
@DeveloperAPI
def collect_episodes(workers: 'WorkerSet', remote_worker_ids: Optional[List[int]]=None, timeout_seconds: int=180) -> List[RolloutMetrics]:
    """Gathers new episodes metrics tuples from the given RolloutWorkers.

    Args:
        workers: WorkerSet.
        remote_worker_ids: Optional list of IDs of remote workers to collect
            metrics from.
        timeout_seconds: Timeout in seconds for collecting metrics from remote workers.

    Returns:
        List of RolloutMetrics.
    """
    metric_lists = workers.foreach_worker(lambda w: w.get_metrics(), local_worker=True, remote_worker_ids=remote_worker_ids, timeout_seconds=timeout_seconds)
    if len(metric_lists) == 0:
        logger.warning('WARNING: collected no metrics.')
    episodes = []
    for metrics in metric_lists:
        episodes.extend(metrics)
    return episodes