import importlib.metadata as importlib_metadata
from typing import Any, Optional
import packaging.version
import torch
from peft.import_utils import is_auto_awq_available
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer
def dispatch_awq(target: torch.nn.Module, adapter_name: str, **kwargs: Any) -> Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if is_auto_awq_available() and isinstance(target_base_layer, WQLinear_GEMM):
        AUTOAWQ_MINIMUM_VERSION = packaging.version.parse('0.2.0')
        version_autoawq = packaging.version.parse(importlib_metadata.version('autoawq'))
        if AUTOAWQ_MINIMUM_VERSION > version_autoawq:
            raise ImportError(f'Found an incompatible version of auto-awq. Found version {version_autoawq}, but only versions above {AUTOAWQ_MINIMUM_VERSION} are supported for PEFT.')
        new_module = AwqLoraLinear(target, adapter_name, **kwargs)
        target.qweight = target_base_layer.qweight
    return new_module