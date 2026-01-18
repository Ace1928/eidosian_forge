from ray.rllib.algorithms.impala.vtrace_tf import VTraceFromLogitsReturns, VTraceReturns
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils import force_list
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
def from_importance_weights(log_rhos, discounts, rewards, values, bootstrap_value, clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):
    """V-trace from log importance weights.

    Calculates V-trace actor critic targets as described in

    "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures"
    by Espeholt, Soyer, Munos et al.

    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size. This code
    also supports the case where all tensors have the same number of additional
    dimensions, e.g., `rewards` is [T, B, C], `values` is [T, B, C],
    `bootstrap_value` is [B, C].

    Args:
        log_rhos: A float32 tensor of shape [T, B] representing the log
            importance sampling weights, i.e.
            log(target_policy(a) / behaviour_policy(a)). V-trace performs
            operations on rhos in log-space for numerical stability.
        discounts: A float32 tensor of shape [T, B] with discounts encountered
            when following the behaviour policy.
        rewards: A float32 tensor of shape [T, B] containing rewards generated
            by following the behaviour policy.
        values: A float32 tensor of shape [T, B] with the value function
            estimates wrt. the target policy.
        bootstrap_value: A float32 of shape [B] with the value function
            estimate at time T.
        clip_rho_threshold: A scalar float32 tensor with the clipping threshold
            for importance weights (rho) when calculating the baseline targets
            (vs). rho^bar in the paper. If None, no clipping is applied.
        clip_pg_rho_threshold: A scalar float32 tensor with the clipping
            threshold on rho_s in
            \\rho_s \\delta log \\pi(a|x) (r + \\gamma v_{s+1} - V(x_s)).
            If None, no clipping is applied.

    Returns:
        A VTraceReturns namedtuple (vs, pg_advantages) where:
        vs: A float32 tensor of shape [T, B]. Can be used as target to
            train a baseline (V(x_t) - vs_t)^2.
        pg_advantages: A float32 tensor of shape [T, B]. Can be used as the
            advantage in the calculation of policy gradients.
    """
    log_rhos = convert_to_torch_tensor(log_rhos, device='cpu')
    discounts = convert_to_torch_tensor(discounts, device='cpu')
    rewards = convert_to_torch_tensor(rewards, device='cpu')
    values = convert_to_torch_tensor(values, device='cpu')
    bootstrap_value = convert_to_torch_tensor(bootstrap_value, device='cpu')
    rho_rank = len(log_rhos.size())
    assert rho_rank == len(values.size())
    assert rho_rank - 1 == len(bootstrap_value.size()), 'must have rank {}'.format(rho_rank - 1)
    assert rho_rank == len(discounts.size())
    assert rho_rank == len(rewards.size())
    rhos = torch.exp(log_rhos)
    if clip_rho_threshold is not None:
        clipped_rhos = torch.clamp_max(rhos, clip_rho_threshold)
    else:
        clipped_rhos = rhos
    cs = torch.clamp_max(rhos, 1.0)
    values_t_plus_1 = torch.cat([values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)
    vs_minus_v_xs = [torch.zeros_like(bootstrap_value)]
    for i in reversed(range(len(discounts))):
        discount_t, c_t, delta_t = (discounts[i], cs[i], deltas[i])
        vs_minus_v_xs.append(delta_t + discount_t * c_t * vs_minus_v_xs[-1])
    vs_minus_v_xs = torch.stack(vs_minus_v_xs[1:])
    vs_minus_v_xs = torch.flip(vs_minus_v_xs, dims=[0])
    vs = vs_minus_v_xs + values
    vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = torch.clamp_max(rhos, clip_pg_rho_threshold)
    else:
        clipped_pg_rhos = rhos
    pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)
    return VTraceReturns(vs=vs.detach(), pg_advantages=pg_advantages.detach())