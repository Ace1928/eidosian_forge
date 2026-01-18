import numpy as np
from typing import Union, Tuple, Any, List
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
@DeveloperAPI
def normc_initializer(std: float=1.0) -> Any:

    def initializer(tensor):
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(tensor.data.pow(2).sum(1, keepdim=True))
    return initializer