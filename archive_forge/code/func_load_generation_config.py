from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.utils.data import Dataset
from .generation.configuration_utils import GenerationConfig
from .integrations.deepspeed import is_deepspeed_zero3_enabled
from .trainer import Trainer
from .utils import logging
@staticmethod
def load_generation_config(gen_config_arg: Union[str, GenerationConfig]) -> GenerationConfig:
    """
        Loads a `~generation.GenerationConfig` from the `Seq2SeqTrainingArguments.generation_config` arguments.

        Args:
            gen_config_arg (`str` or [`~generation.GenerationConfig`]):
                `Seq2SeqTrainingArguments.generation_config` argument.

        Returns:
            A `~generation.GenerationConfig`.
        """
    if isinstance(gen_config_arg, GenerationConfig):
        return deepcopy(gen_config_arg)
    pretrained_model_name = Path(gen_config_arg) if isinstance(gen_config_arg, str) else gen_config_arg
    config_file_name = None
    if pretrained_model_name.is_file():
        config_file_name = pretrained_model_name.name
        pretrained_model_name = pretrained_model_name.parent
    elif pretrained_model_name.is_dir():
        pass
    else:
        pretrained_model_name = gen_config_arg
    gen_config = GenerationConfig.from_pretrained(pretrained_model_name, config_file_name)
    return gen_config