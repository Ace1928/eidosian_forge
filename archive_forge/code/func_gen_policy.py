import argparse
import os
import random
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.examples.models.shared_weights_model import (
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
def gen_policy(i):
    if bool(os.environ.get('RLLIB_ENABLE_RL_MODULE', False)):
        config = {'gamma': random.choice([0.95, 0.99])}
    else:
        config = PPOConfig.overrides(model={'custom_model': ['model1', 'model2'][i % 2]}, gamma=random.choice([0.95, 0.99]))
    return PolicySpec(config=config)