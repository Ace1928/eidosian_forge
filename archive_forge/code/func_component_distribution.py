from typing import Dict
import torch
from torch.distributions import Categorical, constraints
from torch.distributions.distribution import Distribution
@property
def component_distribution(self):
    return self._component_distribution