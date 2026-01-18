import argparse
import numpy as np
from gymnasium.spaces import Discrete
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import (
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.examples.models.centralized_critic_models import (
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_utils import explained_variance, make_tf_callable
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(train_batch[SampleBatch.CUR_OBS], train_batch[OPPONENT_OBS], train_batch[OPPONENT_ACTION])
    policy._central_value_out = model.value_function()
    loss = base_policy.loss(model, dist_class, train_batch)
    model.value_function = vf_saved
    return loss