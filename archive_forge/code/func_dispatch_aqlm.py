from typing import Any, Optional
import torch
from peft.import_utils import is_aqlm_available
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer
def dispatch_aqlm(target: torch.nn.Module, adapter_name: str, **kwargs: Any) -> Optional[torch.nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    if is_aqlm_available() and isinstance(target_base_layer, QuantizedLinear):
        new_module = AqlmLoraLinear(target, adapter_name, **kwargs)
        target.qweight = target_base_layer.codes
    return new_module