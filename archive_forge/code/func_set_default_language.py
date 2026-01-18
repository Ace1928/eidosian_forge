import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_xmod import XmodConfig
def set_default_language(self, language: str):
    """
        Set the default language code for the model. This is used when the language is not specified in the input.

        Args:
            language (`str`): The language code, such as `"en_XX"` or `"de_DE"`.
        """
    if language not in self.config.languages:
        raise ValueError(f'{self} does not have an adapter for {language}. Supported languages: {list(self.config.languages)}')
    self.config.default_language = language