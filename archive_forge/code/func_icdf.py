import math
import torch
from torch import inf
from torch.distributions import constraints
from torch.distributions.cauchy import Cauchy
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AbsTransform
def icdf(self, prob):
    return self.base_dist.icdf((prob + 1) / 2)