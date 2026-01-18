from typing import Dict, List, Type, Union
from ray.rllib.algorithms.marwil.marwil_tf_policy import PostprocessAdvantages
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import ValueNetworkMixin
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import apply_grad_clipping, explained_variance
from ray.rllib.utils.typing import TensorType
@override(TorchPolicyV2)
def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
    stats = {'policy_loss': self.get_tower_stats('p_loss')[0].item(), 'total_loss': self.get_tower_stats('total_loss')[0].item()}
    if self.config['beta'] != 0.0:
        stats['moving_average_sqd_adv_norm'] = self.get_tower_stats('_moving_average_sqd_adv_norm')[0].item()
        stats['vf_explained_var'] = self.get_tower_stats('explained_variance')[0].item()
        stats['vf_loss'] = self.get_tower_stats('v_loss')[0].item()
    return convert_to_numpy(stats)