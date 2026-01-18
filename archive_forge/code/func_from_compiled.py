from typing import Union
import torch
import pytorch_lightning as pl
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0, _TORCH_GREATER_EQUAL_2_1
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy, SingleDeviceStrategy, Strategy
from pytorch_lightning.utilities.model_helpers import _check_mixed_imports
def from_compiled(model: 'torch._dynamo.OptimizedModule') -> 'pl.LightningModule':
    """Returns an instance LightningModule from the output of ``torch.compile``.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    The ``torch.compile`` function returns a ``torch._dynamo.OptimizedModule``, which wraps the LightningModule
    passed in as an argument, but doesn't inherit from it. This means that the output of ``torch.compile`` behaves
    like a LightningModule, but it doesn't inherit from it (i.e. `isinstance` will fail).

    Use this method to obtain a LightningModule that still runs with all the optimizations from ``torch.compile``.

    """
    if not _TORCH_GREATER_EQUAL_2_0:
        raise ModuleNotFoundError('`from_compiled` requires torch>=2.0')
    from torch._dynamo import OptimizedModule
    if not isinstance(model, OptimizedModule):
        raise ValueError(f'`model` is required to be a `OptimizedModule`. Found a `{type(model).__name__}` instead.')
    orig_module = model._orig_mod
    if not isinstance(orig_module, pl.LightningModule):
        _check_mixed_imports(model)
        raise ValueError(f'`model` is expected to be a compiled LightningModule. Found a `{type(orig_module).__name__}` instead')
    orig_module._compiler_ctx = {'compiler': 'dynamo', 'dynamo_ctx': model.dynamo_ctx, 'original_forward': orig_module.forward, 'original_training_step': orig_module.training_step, 'original_validation_step': orig_module.validation_step, 'original_test_step': orig_module.test_step, 'original_predict_step': orig_module.predict_step}
    orig_module.forward = model.dynamo_ctx(orig_module.forward)
    if not _TORCH_GREATER_EQUAL_2_1:
        orig_module.forward._torchdynamo_inline = orig_module.forward
    orig_module.training_step = model.dynamo_ctx(orig_module.training_step)
    if not _TORCH_GREATER_EQUAL_2_1:
        orig_module.training_step._torchdynamo_inline = orig_module.training_step
    orig_module.validation_step = model.dynamo_ctx(orig_module.validation_step)
    orig_module.test_step = model.dynamo_ctx(orig_module.test_step)
    orig_module.predict_step = model.dynamo_ctx(orig_module.predict_step)
    return orig_module