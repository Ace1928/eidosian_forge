from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def set_observation_type(self, observation_type: ObservationType) -> BackendPatternConfig:
    """
        Set how observers should be inserted in the graph for this pattern.

        Observation type here refers to how observers (or quant-dequant ops) will be placed
        in the graph. This is used to produce the desired reference patterns understood by
        the backend. Weighted ops such as linear and conv require different observers
        (or quantization parameters passed to quantize ops in the reference model) for the
        input and the output.

        There are two observation types:

            `OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT` (default): the output observer instance
            will be different from the input. This is the most common observation type.

            `OUTPUT_SHARE_OBSERVER_WITH_INPUT`: the output observer instance will be the
            same as the input. This is useful for operators like `cat`.

        Note: This will be renamed in the near future, since we will soon insert QuantDeQuantStubs
        with observers (and fake quantizes) attached instead of observers themselves.
        """
    self.observation_type = observation_type
    return self