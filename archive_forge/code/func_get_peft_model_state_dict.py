import os
import warnings
from typing import Optional
import torch
from huggingface_hub import file_exists, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.torch import load_file as safe_load_file
from .other import (
from .peft_types import PeftType
def get_peft_model_state_dict(model, state_dict=None, adapter_name='default', unwrap_compiled=False, save_embedding_layers='auto'):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
            the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the passed model will be used.
        adapter_name (`str`, *optional*, defaults to `"default"`):
            The name of the adapter whose state dict should be returned.
        unwrap_compiled (`bool`, *optional*, defaults to `False`):
            Whether to unwrap the model if torch.compile was used.
        save_embedding_layers (`Union[bool, str]`, , *optional*, defaults to `auto`):
            If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common embedding
            layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available. Based on it
            sets the boolean flag. This only works for 🤗 transformers models.
    """
    if unwrap_compiled:
        model = getattr(model, '_orig_mod', model)
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        bias = config.bias
        if bias == 'none':
            to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k}
        elif bias == 'all':
            to_return = {k: state_dict[k] for k in state_dict if 'lora_' in k or 'bias' in k}
        elif bias == 'lora_only':
            to_return = {}
            for k in state_dict:
                if 'lora_' in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split('lora_')[0] + 'bias'
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {k: v for k, v in to_return.items() if 'lora_' in k and adapter_name in k or 'bias' in k}
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {k.replace(f'.{adapter_name}', ''): v for k, v in rank_pattern.items()}
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(rank_pattern, to_return, adapter_name)
    elif config.peft_type == PeftType.LOHA:
        to_return = {k: state_dict[k] for k in state_dict if 'hada_' in k}
    elif config.peft_type == PeftType.LOKR:
        to_return = {k: state_dict[k] for k in state_dict if 'lokr_' in k}
    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {k: state_dict[k] for k in state_dict if k.split('.')[-1].startswith('adaption_')}
    elif config.is_prompt_learning:
        to_return = {}
        if config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            to_return['prefix_task_cols'] = model.prompt_encoder[adapter_name].prefix_task_cols
            to_return['prefix_task_rows'] = model.prompt_encoder[adapter_name].prefix_task_rows
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        elif config.inference_mode:
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return['prompt_embeddings'] = prompt_embeddings
    elif config.peft_type == PeftType.IA3:
        to_return = {k: state_dict[k] for k in state_dict if 'ia3_' in k}
    elif config.peft_type == PeftType.OFT:
        to_return = {k: state_dict[k] for k in state_dict if 'oft_' in k}
    elif config.peft_type == PeftType.POLY:
        to_return = {k: state_dict[k] for k in state_dict if 'poly_' in k}
    else:
        raise NotImplementedError
    if getattr(model, 'modules_to_save', None) is not None:
        for key, value in state_dict.items():
            if any((f'{module_name}.modules_to_save.{adapter_name}' in key for module_name in model.modules_to_save)):
                to_return[key.replace('modules_to_save.', '')] = value
    is_embedding_in_target_modules = False
    if save_embedding_layers == 'auto' and hasattr(config, 'target_modules') and any((k in config.target_modules for k in EMBEDDING_LAYER_NAMES)):
        warnings.warn('Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.')
        save_embedding_layers = is_embedding_in_target_modules = True
    elif save_embedding_layers == 'auto':
        vocab_size = getattr(getattr(model, 'config', None), 'vocab_size', None)
        model_id = getattr(config, 'base_model_name_or_path', None)
        has_remote_config = False
        if model_id is not None:
            exists = check_file_exists_on_hf_hub(model_id, 'config.json')
            if exists is None:
                warnings.warn(f'Could not find a config file in {model_id} - will assume that the vocabulary was not modified.')
                has_remote_config = False
            else:
                has_remote_config = exists
        if vocab_size and model_id and has_remote_config and (vocab_size != model.config.__class__.from_pretrained(model_id).vocab_size):
            warnings.warn('Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.')
            save_embedding_layers = True
        else:
            save_embedding_layers = False
    if save_embedding_layers and hasattr(model, 'get_input_embeddings'):
        for layer in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if not is_embedding_in_target_modules or has_valid_embedding_base_layer(layer):
                embedding_module_name = get_embedding_layer_name(model, layer, is_embedding_in_target_modules)
                if embedding_module_name:
                    to_return.update({k: v for k, v in state_dict.items() if embedding_module_name in k})
    elif save_embedding_layers:
        warnings.warn('Could not identify embedding layer(s) because the model is not a 🤗 transformers model.')
    to_return = {k.replace(f'.{adapter_name}', ''): v for k, v in to_return.items()}
    return to_return