import copy
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any
import ray
from ray.data import Dataset
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, convert_ma_batch_to_sample_batch
from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
@ray.remote
def get_feature_importance_on_index(dataset: ray.data.Dataset, *, index: int, perturb_fn: Callable[[pd.DataFrame, int], None], batch_size: int, policy_state: Dict[str, Any]):
    """A remote function to compute the feature importance of a given index.

    Args:
        dataset: The dataset to use for the computation. The dataset should have `obs`
            and `actions` columns. Each record should be flat d-dimensional array.
        index: The index of the feature to compute the importance for.
        perturb_fn: The function to use for perturbing the dataset at the given index.
        batch_size: The batch size to use for the computation.
        policy_state: The state of the policy to use for the computation.

    Returns:
        The modified dataset that contains a `delta` column which is the absolute
        difference between the expected output and the output due to the perturbation.
    """
    perturbed_ds = dataset.map_batches(perturb_fn, batch_size=batch_size, batch_format='pandas', fn_kwargs={'index': index})
    perturbed_actions = perturbed_ds.map_batches(_compute_actions, batch_size=batch_size, batch_format='pandas', fn_kwargs={'output_key': 'perturbed_actions', 'input_key': 'perturbed_obs', 'policy_state': policy_state})

    def delta_fn(batch):
        batch['delta'] = np.abs(batch['ref_actions'] - batch['perturbed_actions'])
        return batch
    delta = perturbed_actions.map_batches(delta_fn, batch_size=batch_size, batch_format='pandas')
    return delta