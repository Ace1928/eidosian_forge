import gymnasium as gym
from ray.rllib.algorithms.ppo.tf.ppo_tf_rl_module import PPOTfRLModule
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.rllib.utils.torch_utils import FLOAT_MIN
def mask_forward_fn_torch(forward_fn, batch, **kwargs):
    _check_batch(batch)
    action_mask = batch[SampleBatch.OBS]['action_mask']
    batch[SampleBatch.OBS] = batch[SampleBatch.OBS]['observations']
    outputs = forward_fn(batch, **kwargs)
    logits = outputs[SampleBatch.ACTION_DIST_INPUTS]
    inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
    masked_logits = logits + inf_mask
    outputs[SampleBatch.ACTION_DIST_INPUTS] = masked_logits
    return outputs