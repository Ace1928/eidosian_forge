import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Union
import torch
from xformers.components import Activation
from xformers.components.feedforward import (
def expert_constructor() -> torch.nn.Module:
    return MLP(dim_model, dropout, activation, multiplier)