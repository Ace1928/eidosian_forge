import gymnasium as gym
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import (
from ray.rllib.utils.typing import (
from ray.util import log_once
@override(BaseEnv)
def send_actions(self, action_dict: MultiEnvDict) -> None:
    for env_id, agent_dict in action_dict.items():
        if env_id in self.terminateds or env_id in self.truncateds:
            raise ValueError(f'Env {env_id} is already done and cannot accept new actions')
        env = self.envs[env_id]
        try:
            obs, rewards, terminateds, truncateds, infos = env.step(agent_dict)
        except Exception as e:
            if self.restart_failed_sub_environments:
                logger.exception(e.args[0])
                self.try_restart(env_id=env_id)
                obs = e
                rewards = {}
                terminateds = {'__all__': True}
                truncateds = {'__all__': False}
                infos = {}
            else:
                raise e
        assert isinstance(obs, (dict, Exception)), 'Not a multi-agent obs dict or an Exception!'
        assert isinstance(rewards, dict), 'Not a multi-agent reward dict!'
        assert isinstance(terminateds, dict), 'Not a multi-agent terminateds dict!'
        assert isinstance(truncateds, dict), 'Not a multi-agent truncateds dict!'
        assert isinstance(infos, dict), 'Not a multi-agent info dict!'
        if isinstance(obs, dict):
            info_diff = set(infos).difference(set(obs))
            if info_diff and info_diff != {'__common__'}:
                raise ValueError("Key set for infos must be a subset of obs (plus optionally the '__common__' key for infos concerning all/no agents): {} vs {}".format(infos.keys(), obs.keys()))
        if '__all__' not in terminateds:
            raise ValueError("In multi-agent environments, '__all__': True|False must be included in the 'terminateds' dict: got {}.".format(terminateds))
        elif '__all__' not in truncateds:
            raise ValueError("In multi-agent environments, '__all__': True|False must be included in the 'truncateds' dict: got {}.".format(truncateds))
        if terminateds['__all__']:
            self.terminateds.add(env_id)
        if truncateds['__all__']:
            self.truncateds.add(env_id)
        self.env_states[env_id].observe(obs, rewards, terminateds, truncateds, infos)