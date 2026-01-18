from gymnasium.spaces import Dict, Discrete, MultiDiscrete, Tuple
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
def agent_1_obs(self):
    if self.one_hot_state_encoding:
        return np.concatenate([self.state, [1]])
    else:
        return np.flatnonzero(self.state)[0]