import abc
from typing import List, Optional, Tuple, Union
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.specs.specs_base import Spec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.rnn_sequencing import get_fold_unfold_fns
from ray.rllib.utils.annotations import ExperimentalAPI, DeveloperAPI
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
@input_specs.setter
def input_specs(self, spec: Spec) -> None:
    raise ValueError('`input_specs` cannot be set directly. Override Model.get_input_specs() instead. Set Model._input_specs if you want to override this behavior.')