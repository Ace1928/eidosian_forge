from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from accelerate import init_empty_weights
from accelerate.commands.utils import CustomArgumentParser
from accelerate.utils import (
def estimate_training_usage(bytes: int, mixed_precision: str, msamp_config: str=None) -> dict:
    """
    Given an amount of `bytes` and `mixed_precision`, calculates how much training memory is needed for a batch size of
    1.

    Args:
        bytes (`int`):
            The size of the model being trained.
        mixed_precision (`str`):
            The mixed precision that would be ran.
        msamp_config (`str`):
            The msamp config to estimate the training memory for if `mixed_precision` is set to `"fp8"`.
    """
    memory_sizes = {'model': -1, 'optimizer': -1, 'gradients': -1, 'step': -1}
    fp32_size = bytes
    fp16_size = bytes // 2
    if mixed_precision == 'float32':
        memory_sizes['model'] = fp32_size
        memory_sizes['gradients'] = fp32_size
        memory_sizes['optimizer'] = fp32_size * 2
        memory_sizes['step'] = fp32_size * 4
    elif mixed_precision in ('float16', 'bfloat16') or (mixed_precision == 'fp8' and msamp_config is None):
        memory_sizes['model'] = fp32_size
        memory_sizes['gradients'] = fp32_size + fp16_size
        memory_sizes['optimizer'] = fp32_size * 2
        memory_sizes['step'] = memory_sizes['optimizer']
    return memory_sizes