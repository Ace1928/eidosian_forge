from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f'--> applying fsdp activation checkpointing...')
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)