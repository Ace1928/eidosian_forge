from ray.rllib.algorithms.impala.vtrace_tf import VTraceFromLogitsReturns, VTraceReturns
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils import force_list
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
def get_log_rhos(target_action_log_probs, behaviour_action_log_probs):
    """With the selected log_probs for multi-discrete actions of behavior
    and target policies we compute the log_rhos for calculating the vtrace."""
    t = torch.stack(target_action_log_probs)
    b = torch.stack(behaviour_action_log_probs)
    log_rhos = torch.sum(t - b, dim=0)
    return log_rhos