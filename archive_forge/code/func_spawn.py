import gymnasium as gym
import numpy as np
import random
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.rllib.examples.env.mock_env import MockEnv, MockEnv2
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
from ray.rllib.utils.deprecation import Deprecated
def spawn(self):
    agentID = self.agentID
    self.agents[agentID] = MockEnv(25)
    self._agent_ids.add(agentID)
    self.agentID += 1
    return agentID