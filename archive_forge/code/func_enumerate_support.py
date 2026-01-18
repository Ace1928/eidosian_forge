import torch
from torch import nan
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property, logits_to_probs, probs_to_logits
def enumerate_support(self, expand=True):
    num_events = self._num_events
    values = torch.arange(num_events, dtype=torch.long, device=self._param.device)
    values = values.view((-1,) + (1,) * len(self._batch_shape))
    if expand:
        values = values.expand((-1,) + self._batch_shape)
    return values