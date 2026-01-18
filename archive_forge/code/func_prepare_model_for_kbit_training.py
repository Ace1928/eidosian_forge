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
def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
    """
    Note this method only works for `transformers` models.

    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
        use_gradient_checkpointing (`bool`, *optional*, defaults to `True`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        gradient_checkpointing_kwargs (`dict`, *optional*, defaults to `None`):
            Keyword arguments to pass to the gradient checkpointing function, please refer to the documentation of
            `torch.utils.checkpoint.checkpoint` for more details about the arguments that you can pass to that method.
            Note this is only available in the latest transformers versions (> 4.34.1).
    """
    loaded_in_kbit = getattr(model, 'is_loaded_in_8bit', False) or getattr(model, 'is_loaded_in_4bit', False)
    is_gptq_quantized = getattr(model, 'quantization_method', None) == 'gptq'
    is_aqlm_quantized = getattr(model, 'quantization_method', None) == 'aqlm'
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}
    for name, param in model.named_parameters():
        param.requires_grad = False
    if not is_gptq_quantized and (not is_aqlm_quantized):
        for param in model.parameters():
            if param.dtype == torch.float16 or param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float32)
    if (loaded_in_kbit or is_gptq_quantized or is_aqlm_quantized) and use_gradient_checkpointing:
        if 'use_reentrant' not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs['use_reentrant']:
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        _supports_gc_kwargs = 'gradient_checkpointing_kwargs' in list(inspect.signature(model.gradient_checkpointing_enable).parameters)
        if not _supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
            warnings.warn('gradient_checkpointing_kwargs is not supported in this version of transformers. The passed kwargs will be ignored. if you want to use that feature, please upgrade to the latest version of transformers.', FutureWarning)
        gc_enable_kwargs = {} if not _supports_gc_kwargs else {'gradient_checkpointing_kwargs': gradient_checkpointing_kwargs}
        model.gradient_checkpointing_enable(**gc_enable_kwargs)
    return model