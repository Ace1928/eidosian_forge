import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Union
from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin
from .utils import CONFIG_NAME, PeftType, TaskType
@classmethod
def from_peft_type(cls, **kwargs):
    """
        This method loads the configuration of your adapter model from a set of kwargs.

        The appropriate configuration type is determined by the `peft_type` argument. If `peft_type` is not provided,
        the calling class type is instantiated.

        Args:
            kwargs (configuration keyword arguments):
                Keyword arguments passed along to the configuration initialization.
        """
    from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
    if 'peft_type' in kwargs:
        peft_type = kwargs['peft_type']
        config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]
    else:
        config_cls = cls
    return config_cls(**kwargs)