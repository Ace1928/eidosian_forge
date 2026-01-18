from __future__ import annotations
from typing import TYPE_CHECKING, Any
import torch
from .config import PeftConfig
from .mixed_model import PeftMixedModel
from .peft_model import (
from .tuners import (
from .utils import _prepare_prompt_learning_config
def inject_adapter_in_model(peft_config: PeftConfig, model: torch.nn.Module, adapter_name: str='default') -> torch.nn.Module:
    """
    A simple API to create and inject adapter in-place into a model. Currently the API does not support prompt learning
    methods and adaption prompt. Make sure to have the correct `target_names` set in the `peft_config` object. The API
    calls `get_peft_model` under the hood but would be restricted only to non-prompt learning methods.

    Args:
        peft_config (`PeftConfig`):
            Configuration object containing the parameters of the Peft model.
        model (`torch.nn.Module`):
            The input model where the adapter will be injected.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
    """
    if peft_config.is_prompt_learning or peft_config.is_adaption_prompt:
        raise ValueError('`create_and_replace` does not support prompt learning and adaption prompt yet.')
    if peft_config.peft_type not in PEFT_TYPE_TO_TUNER_MAPPING.keys():
        raise ValueError(f'`inject_adapter_in_model` does not support {peft_config.peft_type} yet. Please use `get_peft_model`.')
    tuner_cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
    peft_model = tuner_cls(model, peft_config, adapter_name=adapter_name)
    return peft_model.model