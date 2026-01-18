import argparse
import os
import sys
import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.utils import try_import_pyspiel, try_import_open_spiel
from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.tune import CLIReporter, register_env
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.examples.rl_module.random_rl_module import RandomRLModule
from open_spiel.python.rl_environment import Environment  # noqa: E402
def ask_user_for_action(time_step):
    """Asks the user for a valid action on the command line and returns it.

    Re-queries the user until she picks a valid one.

    Args:
        time_step: The open spiel Environment time-step object.
    """
    pid = time_step.observations['current_player']
    legal_moves = time_step.observations['legal_actions'][pid]
    choice = -1
    while choice not in legal_moves:
        print('Choose an action from {}:'.format(legal_moves))
        sys.stdout.flush()
        choice_str = input()
        try:
            choice = int(choice_str)
        except ValueError:
            continue
    return choice