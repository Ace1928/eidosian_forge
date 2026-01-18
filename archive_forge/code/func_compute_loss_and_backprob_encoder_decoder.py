import timeit
from typing import Callable, Optional
from ..configuration_utils import PretrainedConfig
from ..models.auto.modeling_auto import MODEL_MAPPING, MODEL_WITH_LM_HEAD_MAPPING
from ..utils import is_py3nvml_available, is_torch_available, logging
from .benchmark_utils import (
def compute_loss_and_backprob_encoder_decoder():
    loss = train_model(input_ids, decoder_input_ids=input_ids, labels=input_ids)[0]
    loss.backward()
    return loss