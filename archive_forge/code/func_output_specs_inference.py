from typing import Mapping, Any
from ray.rllib.core.rl_module import RLModule
from ray.rllib.policy.sample_batch import SampleBatch
import tree
import pathlib
import gymnasium as gym
def output_specs_inference(self):
    return [SampleBatch.ACTIONS]