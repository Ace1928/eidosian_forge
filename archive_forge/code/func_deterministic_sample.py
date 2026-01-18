from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
def deterministic_sample(self):
    a1_dist = self._a1_distribution()
    a1 = a1_dist.deterministic_sample()
    a2_dist = self._a2_distribution(a1)
    a2 = a2_dist.deterministic_sample()
    self._action_logp = a1_dist.logp(a1) + a2_dist.logp(a2)
    return (a1, a2)