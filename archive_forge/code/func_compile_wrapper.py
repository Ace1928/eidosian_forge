import pathlib
from typing import Any, List, Mapping, Tuple, Union, Type
from packaging import version
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.rl_module_with_target_networks_interface import (
from ray.rllib.core.rl_module.torch.torch_compile_config import TorchCompileConfig
from ray.rllib.models.torch.torch_distributions import TorchDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import TORCH_COMPILE_REQUIRED_VERSION
from ray.rllib.utils.typing import NetworkType
def compile_wrapper(rl_module: 'TorchRLModule', compile_config: TorchCompileConfig):
    """A wrapper that compiles the forward methods of a TorchRLModule."""
    if torch is not None and version.parse(torch.__version__) < TORCH_COMPILE_REQUIRED_VERSION:
        raise ValueError('torch.compile is only supported from torch 2.0.0')
    compiled_forward_train = torch.compile(rl_module._forward_train, backend=compile_config.torch_dynamo_backend, mode=compile_config.torch_dynamo_mode, **compile_config.kwargs)
    rl_module._forward_train = compiled_forward_train
    compiled_forward_inference = torch.compile(rl_module._forward_inference, backend=compile_config.torch_dynamo_backend, mode=compile_config.torch_dynamo_mode, **compile_config.kwargs)
    rl_module._forward_inference = compiled_forward_inference
    compiled_forward_exploration = torch.compile(rl_module._forward_exploration, backend=compile_config.torch_dynamo_backend, mode=compile_config.torch_dynamo_mode, **compile_config.kwargs)
    rl_module._forward_exploration = compiled_forward_exploration
    return rl_module