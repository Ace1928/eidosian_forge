import os
from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type, Union
import transformers
from .. import PretrainedConfig, is_tf_available, is_torch_available
from ..utils import TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from .config import OnnxConfig
@staticmethod
def check_supported_model_or_raise(model: Union['PreTrainedModel', 'TFPreTrainedModel'], feature: str='default') -> Tuple[str, Callable]:
    """
        Check whether or not the model has the requested features.

        Args:
            model: The model to export.
            feature: The name of the feature to check if it is available.

        Returns:
            (str) The type of the model (OnnxConfig) The OnnxConfig instance holding the model export properties.

        """
    model_type = model.config.model_type.replace('_', '-')
    model_name = getattr(model, 'name', '')
    model_features = FeaturesManager.get_supported_features_for_model_type(model_type, model_name=model_name)
    if feature not in model_features:
        raise ValueError(f"{model.config.model_type} doesn't support feature {feature}. Supported values are: {model_features}")
    return (model.config.model_type, FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][feature])