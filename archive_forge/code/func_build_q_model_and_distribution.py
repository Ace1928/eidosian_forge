from typing import Dict, List, Tuple
import gymnasium as gym
import ray
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.exploration.parameter_noise import ParameterNoise
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import (
from ray.rllib.utils.typing import TensorType, AlgorithmConfigDict
def build_q_model_and_distribution(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> Tuple[ModelV2, TorchDistributionWrapper]:
    """Build q_model and target_model for DQN

    Args:
        policy: The policy, which will use the model for optimization.
        obs_space (gym.spaces.Space): The policy's observation space.
        action_space (gym.spaces.Space): The policy's action space.
        config (AlgorithmConfigDict):

    Returns:
        (q_model, TorchCategorical)
            Note: The target q model will not be returned, just assigned to
            `policy.target_model`.
    """
    if not isinstance(action_space, gym.spaces.Discrete):
        raise UnsupportedSpaceException('Action space {} is not supported for DQN.'.format(action_space))
    if config['hiddens']:
        num_outputs = ([256] + list(config['model']['fcnet_hiddens']))[-1]
        config['model']['no_final_linear'] = True
    else:
        num_outputs = action_space.n
    add_layer_norm = isinstance(getattr(policy, 'exploration', None), ParameterNoise) or config['exploration_config']['type'] == 'ParameterNoise'
    model = ModelCatalog.get_model_v2(obs_space=obs_space, action_space=action_space, num_outputs=num_outputs, model_config=config['model'], framework='torch', model_interface=DQNTorchModel, name=Q_SCOPE, q_hiddens=config['hiddens'], dueling=config['dueling'], num_atoms=config['num_atoms'], use_noisy=config['noisy'], v_min=config['v_min'], v_max=config['v_max'], sigma0=config['sigma0'], add_layer_norm=add_layer_norm)
    policy.target_model = ModelCatalog.get_model_v2(obs_space=obs_space, action_space=action_space, num_outputs=num_outputs, model_config=config['model'], framework='torch', model_interface=DQNTorchModel, name=Q_TARGET_SCOPE, q_hiddens=config['hiddens'], dueling=config['dueling'], num_atoms=config['num_atoms'], use_noisy=config['noisy'], v_min=config['v_min'], v_max=config['v_max'], sigma0=config['sigma0'], add_layer_norm=add_layer_norm)
    temperature = config['categorical_distribution_temperature']
    return (model, get_torch_categorical_class_with_temperature(temperature))