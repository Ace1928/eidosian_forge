from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from ..utils import is_torch_available
from ..utils.quantization_config import QuantizationConfigMixin
def preprocess_model(self, model: 'PreTrainedModel', **kwargs):
    """
        Setting model attributes and/or converting model before weights loading. At this point
        the model should be initialized on the meta device so you can freely manipulate the skeleton
        of the model in order to replace modules in-place. Make sure to override the abstract method `_process_model_before_weight_loading`.

        Args:
            model (`~transformers.PreTrainedModel`):
                The model to quantize
            kwargs (`dict`, *optional*):
                The keyword arguments that are passed along `_process_model_before_weight_loading`.
        """
    model.is_quantized = True
    model.quantization_method = self.quantization_config.quant_method
    return self._process_model_before_weight_loading(model, **kwargs)