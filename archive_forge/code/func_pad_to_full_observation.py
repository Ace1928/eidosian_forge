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
def pad_to_full_observation(self, hidden_states):
    batch_size, sequence_length, _ = hidden_states.shape
    n_pad = (self.transition_dim - sequence_length % self.transition_dim) % self.transition_dim
    padding = torch.zeros(batch_size, n_pad, self.embedding_dim, device=hidden_states.device)
    hidden_states_pad = torch.cat([hidden_states, padding], dim=1)
    hidden_states_pad = hidden_states_pad.view(-1, self.transition_dim, self.embedding_dim)
    return (hidden_states_pad, n_pad)