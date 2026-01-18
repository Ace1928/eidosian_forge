import os
import warnings
from typing import Optional
import torch
from huggingface_hub import file_exists, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file as safe_load_file
from .other import (
from .peft_types import PeftType
def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name='default'):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if getattr(model, 'modules_to_save', None) is not None:
        for key, value in peft_model_state_dict.items():
            if any((module_name in key for module_name in model.modules_to_save)):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f'{module_name}.modules_to_save.{adapter_name}')
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict
    if config.peft_type in (PeftType.LORA, PeftType.LOHA, PeftType.LOKR, PeftType.ADALORA, PeftType.IA3, PeftType.OFT, PeftType.POLY):
        peft_model_state_dict = {}
        parameter_prefix = {PeftType.IA3: 'ia3_', PeftType.LORA: 'lora_', PeftType.ADALORA: 'lora_', PeftType.LOHA: 'hada_', PeftType.LOKR: 'lokr_', PeftType.OFT: 'oft_', PeftType.POLY: 'poly_'}[config.peft_type]
        for k, v in state_dict.items():
            if parameter_prefix in k:
                suffix = k.split(parameter_prefix)[1]
                if '.' in suffix:
                    suffix_to_replace = '.'.join(suffix.split('.')[1:])
                    k = k.replace(suffix_to_replace, f'{adapter_name}.{suffix_to_replace}')
                else:
                    k = f'{k}.{adapter_name}'
                peft_model_state_dict[k] = v
            else:
                peft_model_state_dict[k] = v
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
    elif config.is_prompt_learning or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError
    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    if config.is_prompt_learning:
        model.prompt_encoder[adapter_name].embedding.load_state_dict({'weight': peft_model_state_dict['prompt_embeddings']}, strict=True)
    if config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
        model.prompt_encoder[adapter_name].load_state_dict(peft_model_state_dict, strict=False)
    return load_result