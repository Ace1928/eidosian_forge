from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from ..utils import is_torch_available
from ..utils.quantization_config import QuantizationConfigMixin
def get_special_dtypes_update(self, model, torch_dtype: 'torch.dtype') -> Dict[str, 'torch.dtype']:
    """
        returns dtypes for modules that are not quantized - used for the computation of the device_map in case
        one passes a str as a device_map. The method will use the `modules_to_not_convert` that is modified
        in `_process_model_before_weight_loading`.

        Args:
            model (`~transformers.PreTrainedModel`):
                The model to quantize
            torch_dtype (`torch.dtype`):
                The dtype passed in `from_pretrained` method.
        """
    return {name: torch_dtype for name, _ in model.named_parameters() if any((m in name for m in self.modules_to_not_convert))}