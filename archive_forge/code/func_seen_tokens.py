from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from .configuration_utils import PretrainedConfig
from .utils import logging
@property
def seen_tokens(self):
    logger.warning_once('The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` model input instead.')
    if hasattr(self, '_seen_tokens'):
        return self._seen_tokens
    else:
        return None