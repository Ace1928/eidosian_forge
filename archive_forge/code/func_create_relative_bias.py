import collections
import logging
import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import UdopConfig
from transformers.modeling_outputs import (
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ..deprecated._archive_maps import UDOP_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
def create_relative_bias(config: UdopConfig) -> Sequence[RelativePositionBiasBase]:
    """
    Creates empty list or one/multiple relative biases.

    :param config: Model's configuration :return: Sequence with created bias modules.
    """
    bias_list = []
    if hasattr(config, 'relative_bias_args'):
        for bias_kwargs_org in config.relative_bias_args:
            bias_kwargs = deepcopy(bias_kwargs_org)
            bias_type = bias_kwargs.pop('type')
            model_num_heads = config.num_heads if hasattr(config, 'num_heads') else config.num_attention_heads
            if 'num_heads' in bias_kwargs:
                if bias_kwargs['num_heads'] != model_num_heads:
                    raise ValueError('Number of heads must match num of heads in the model')
            else:
                bias_kwargs['num_heads'] = model_num_heads
            bias_list.append(BIAS_CLASSES[bias_type](**bias_kwargs))
    return bias_list