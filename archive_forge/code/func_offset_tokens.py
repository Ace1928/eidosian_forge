import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ....modeling_utils import PreTrainedModel
from ....utils import (
from .configuration_trajectory_transformer import TrajectoryTransformerConfig
def offset_tokens(self, trajectories):
    _, sequence_length = trajectories.shape
    n_states = int(np.ceil(sequence_length / self.transition_dim))
    offsets = torch.arange(self.transition_dim) * self.vocab_size
    offsets = offsets.repeat(n_states).to(trajectories.device)
    offset_trajectories = trajectories + offsets[:sequence_length]
    offset_trajectories[trajectories == self.vocab_size] = self.stop_token
    return offset_trajectories