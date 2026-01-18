import inspect
import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar, Union, get_args
from .constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME, SAFETENSORS_SINGLE_FILE
from .file_download import hf_hub_download
from .hf_api import HfApi
from .repocard import ModelCard, ModelCardData
from .utils import (
from .utils._deprecation import _deprecate_arguments
def generate_model_card(self, *args, **kwargs) -> ModelCard:
    card = ModelCard.from_template(card_data=ModelCardData(**asdict(self._hub_mixin_info)), template_str=DEFAULT_MODEL_CARD)
    return card