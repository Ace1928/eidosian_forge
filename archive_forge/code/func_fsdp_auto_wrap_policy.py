import copy
import inspect
import os
import warnings
from contextlib import nullcontext
from typing import Optional, Tuple
import accelerate
import torch
from accelerate.hooks import add_hook_to_module, remove_hook_from_module
from accelerate.utils import is_npu_available, is_xpu_available
from huggingface_hub import file_exists
from huggingface_hub.utils import EntryNotFoundError, HFValidationError
from safetensors.torch import storage_ptr, storage_size
from ..import_utils import is_auto_gptq_available, is_torch_tpu_available
from .constants import (
def fsdp_auto_wrap_policy(model):
    import functools
    import os
    from accelerate import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
    from ..tuners import PrefixEncoder, PromptEmbedding, PromptEncoder
    default_transformer_cls_names_to_wrap = ','.join(model._no_split_modules) if getattr(model, '_no_split_modules', None) is not None else ''
    transformer_cls_names_to_wrap = os.environ.get('FSDP_TRANSFORMER_CLS_TO_WRAP', default_transformer_cls_names_to_wrap).split(',')
    transformer_cls_to_wrap = {PrefixEncoder, PromptEncoder, PromptEmbedding}
    for layer_class in transformer_cls_names_to_wrap:
        transformer_cls = FullyShardedDataParallelPlugin.get_module_class_from_name(model, layer_class)
        if transformer_cls is None:
            raise Exception('Could not find the transformer layer class to wrap in the model.')
        else:
            transformer_cls_to_wrap.add(transformer_cls)

    def lambda_policy_fn(module):
        if len(list(module.named_children())) == 0 and getattr(module, 'weight', None) is not None and module.weight.requires_grad:
            return True
        return False
    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_cls_to_wrap)
    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy