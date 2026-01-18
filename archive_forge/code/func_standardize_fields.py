import logging
from typing import List, Optional, Union
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import (
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.sgd import standardized
from ray.rllib.utils.typing import SampleBatchType
def standardize_fields(samples: SampleBatchType, fields: List[str]) -> SampleBatchType:
    """Standardize fields of the given SampleBatch"""
    wrapped = False
    if isinstance(samples, SampleBatch):
        samples = samples.as_multi_agent()
        wrapped = True
    for policy_id in samples.policy_batches:
        batch = samples.policy_batches[policy_id]
        for field in fields:
            if field in batch:
                batch[field] = standardized(batch[field])
    if wrapped:
        samples = samples.policy_batches[DEFAULT_POLICY_ID]
    return samples