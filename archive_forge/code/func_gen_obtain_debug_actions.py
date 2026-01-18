from minerl.env.malmo import InstanceManager
import minerl
import time
import gym
import numpy as np
import logging
import coloredlogs
from minerl.herobraine.wrappers.vector_wrapper import Vectorized
from minerl.herobraine.env_specs.obtain_specs import ObtainDiamondDebug
from minerl.herobraine.hero.test_spaces import assert_equal_recursive
from minerl.herobraine.wrappers.obfuscation_wrapper import Obfuscated
import minerl.herobraine.envs as envs
import minerl.herobraine
def gen_obtain_debug_actions(env):
    actions = []

    def act(**kwargs):
        action = env.action_space.no_op()
        for key, value in kwargs.items():
            action[key] = value
        actions.append(action)
    act(camera=np.array([45.0, 0.0], dtype=np.float32))
    act(place='log')
    act()
    act(craft='stick')
    act(craft='stick')
    act(craft='planks')
    act(craft='crafting_table')
    act(camera=np.array([0.0, 90.0], dtype=np.float32))
    act(nearbyCraft='stone_pickaxe')
    act(place='crafting_table')
    act(nearbyCraft='stone_pickaxe')
    act(nearbyCraft='furnace')
    act(camera=np.array([0.0, 90.0], dtype=np.float32))
    act(nearbySmelt='iron_ingot')
    act(place='furnace')
    act(nearbySmelt='iron_ingot')
    act(nearbySmelt='iron_ingot')
    act(nearbySmelt='iron_ingot')
    act(camera=np.array([45.0, 0.0], dtype=np.float32))
    act(attack=1)
    [(act(jump=1), act(jump=1), act(jump=1), act(jump=1), act(jump=1), act(place='cobblestone'), act()) for _ in range(2)]
    act(equip='stone_pickaxe')
    [act(attack=1) for _ in range(40)]
    act(camera=np.array([-45.0, -90.0], dtype=np.float32))
    act(nearbyCraft='stone_axe')
    act(equip='stone_axe')
    for _ in range(2):
        [act(attack=1) for _ in range(30)]
        [act(forward=1) for _ in range(10)]
        [act(back=1) for _ in range(10)]
        act(place='crafting_table')
    act(camera=np.array([0.0, -90.0], dtype=np.float32))
    [act(attack=1) for _ in range(20)]
    [act(forward=1) for _ in range(10)]
    act(equip='air')
    [act(attack=1) for _ in range(62)]
    [act(forward=1) for _ in range(10)]
    act(craft='planks')
    act(craft='crafting_table')
    act(place='crafting_table')
    act(camera=np.array([0.0, -90.0], dtype=np.float32))
    act(camera=np.array([0.0, -90.0], dtype=np.float32))
    act(craft='planks')
    act(craft='stick')
    act(craft='stick')
    act(nearbyCraft='iron_pickaxe')
    act(equip='iron_pickaxe')
    for _ in range(2):
        act(place='diamond_ore')
        [act(attack=1) for _ in range(20)]
        [act(forward=1) for _ in range(10)]
    [act() for _ in range(10)]
    return actions