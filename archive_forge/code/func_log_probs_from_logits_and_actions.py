from ray.rllib.algorithms.impala.vtrace_tf import VTraceFromLogitsReturns, VTraceReturns
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils import force_list
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
def log_probs_from_logits_and_actions(policy_logits, actions, dist_class=TorchCategorical, model=None):
    return multi_log_probs_from_logits_and_actions([policy_logits], [actions], dist_class, model)[0]