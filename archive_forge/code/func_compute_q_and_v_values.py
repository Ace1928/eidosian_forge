import numpy as np
import pandas as pd
from typing import Any, Dict, Type, TYPE_CHECKING
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.annotations import DeveloperAPI
@DeveloperAPI
def compute_q_and_v_values(batch: pd.DataFrame, model_class: Type['FQETorchModel'], model_state: Dict[str, Any], compute_q_values: bool=True) -> pd.DataFrame:
    """Computes the Q and V values for the given batch of samples.

    This function is to be used with map_batches() to perform a batch prediction on a
    dataset of records with `obs` and `actions` columns.

    Args:
        batch: A sub-batch from the dataset.
        model_class: The model class to use for the prediction. This class should be a
            sub-class of FQEModel that implements the estimate_q() and estimate_v()
            methods.
        model_state: The state of the model to use for the prediction.
        compute_q_values: Whether to compute the Q values or not. If False, only the V
            is computed and returned.

    Returns:
        The modified batch with the Q and V values added as columns.
    """
    model = model_class.from_state(model_state)
    sample_batch = SampleBatch({SampleBatch.OBS: np.vstack(batch[SampleBatch.OBS]), SampleBatch.ACTIONS: np.vstack(batch[SampleBatch.ACTIONS]).squeeze(-1)})
    v_values = model.estimate_v(sample_batch)
    v_values = convert_to_numpy(v_values)
    batch['v_values'] = v_values
    if compute_q_values:
        q_values = model.estimate_q(sample_batch)
        q_values = convert_to_numpy(q_values)
        batch['q_values'] = q_values
    return batch