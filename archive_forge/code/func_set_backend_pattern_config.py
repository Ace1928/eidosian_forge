from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def set_backend_pattern_config(self, config: BackendPatternConfig) -> BackendConfig:
    """
        Set the config for an pattern that can be run on the target backend.
        This overrides any existing config for the given pattern.
        """
    pattern_complex_format = torch.ao.quantization.backend_config.utils._get_pattern_in_reversed_nested_tuple_format(config)
    self._pattern_complex_format_to_config[pattern_complex_format] = config
    return self