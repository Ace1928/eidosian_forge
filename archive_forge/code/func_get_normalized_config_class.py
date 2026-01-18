import functools
from typing import Callable, Dict, Type, Union
from transformers import PretrainedConfig
@classmethod
def get_normalized_config_class(cls, model_type: str) -> Type:
    model_type = model_type.replace('_', '-')
    cls.check_supported_model(model_type)
    return cls._conf[model_type]