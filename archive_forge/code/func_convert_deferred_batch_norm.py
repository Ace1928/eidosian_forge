from typing import TypeVar, cast
import torch
from torch import Tensor, nn
from torch.nn.functional import batch_norm
from torch.nn.modules.batchnorm import _BatchNorm
from .checkpoint import is_recomputing
@classmethod
def convert_deferred_batch_norm(cls, module: TModule, chunks: int=1) -> TModule:
    """Converts a :class:`nn.BatchNorm` or underlying :class:`nn.BatchNorm`s into :class:`DeferredBatchNorm`::

            from torchvision.models.resnet import resnet101
            from torchpipe.batchnorm import DeferredBatchNorm
            model = resnet101()
            model = DeferredBatchNorm.convert_deferred_batch_norm(model)

        """
    if isinstance(module, DeferredBatchNorm) and module.chunks is chunks:
        return cast(TModule, module)
    module_output: nn.Module = module
    if isinstance(module, _BatchNorm) and module.track_running_stats:
        module_output = DeferredBatchNorm(module.num_features, module.eps, module.momentum, module.affine, chunks)
        if module.affine:
            module_output.register_parameter('weight', module.weight)
            module_output.register_parameter('bias', module.bias)
        module_output.register_buffer('running_mean', module.running_mean)
        module_output.register_buffer('running_var', module.running_var)
        module_output.register_buffer('num_batches_tracked', module.num_batches_tracked)
    for name, child in module.named_children():
        module_output.add_module(name, cls.convert_deferred_batch_norm(child, chunks))
    return cast(TModule, module_output)