from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def set_backend_pattern_configs(self, configs: List[BackendPatternConfig]) -> BackendConfig:
    """
        Set the configs for patterns that can be run on the target backend.
        This overrides any existing config for a given pattern if it was previously registered already.
        """
    for conf in configs:
        self.set_backend_pattern_config(conf)
    return self