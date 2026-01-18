from minerl.herobraine.hero import spaces
import os
from minerl.herobraine.hero.test_spaces import assert_equal_recursive
import numpy as np
import gym
import xmltodict
def test_env_space_regressions():
    import minerl.herobraine.env_specs
    for env in old_envs:
        newspec = gym.envs.registration.spec(env['id'])
        k1 = newspec._kwargs
        k2 = env['kwargs']
        task = k1['env_spec']
        assert task.action_space == env['kwargs']['action_space']
        assert task.observation_space == env['kwargs']['observation_space']
        assert newspec.max_episode_steps == env['max_episode_steps']
        if 'reward_threshold' in env or hasattr(newspec, 'reward_threshold'):
            assert newspec.reward_threshold == env['reward_threshold']
        with open(env['kwargs']['xml'], 'rt') as f:
            old_env_xml = f.read()
        new_env_xml = newspec._kwargs['env_spec'].to_xml()
        old_xml_dict = xmltodict.parse(old_env_xml)
        new_xml_dict = xmltodict.parse(new_env_xml)
        assert_equal_recursive(new_xml_dict, old_xml_dict, ignore=['@generatorOptions', 'Name', 'About'])