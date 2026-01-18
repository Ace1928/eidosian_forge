import functools
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp.wrap import (
def get_llama_wrapper():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    llama_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
    return llama_auto_wrap_policy