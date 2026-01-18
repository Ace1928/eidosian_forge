from contextlib import contextmanager
import packaging.version
import torch
import transformers
@contextmanager
def gather_params_ctx(module: torch.nn.Module, modifier_rank: int=0):
    """Call DeepSpeed GatheredParameters context manager if DeepSpeed is enabled, otherwise do nothing."""
    if packaging.version.parse(transformers.__version__) >= packaging.version.parse('4.33.0'):
        from transformers.integrations import is_deepspeed_zero3_enabled
    else:
        from transformers.deepspeed import is_deepspeed_zero3_enabled
    if not is_deepspeed_zero3_enabled():
        yield
        return
    import deepspeed
    params_to_gather = module.parameters()
    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=modifier_rank):
        yield
    return