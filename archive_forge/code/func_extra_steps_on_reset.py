import gym
import json
import numpy as np
from copy import deepcopy
from minerl.herobraine.hero import mc, handlers
from collections import defaultdict, deque
def extra_steps_on_reset(self, ob):
    for i in range(len(self.actions) - 1):
        a, na = (self.actions[i], self.actions[i + 1])
        sprint_stat = 'minecraft.custom:minecraft.sprint_one_cm'
        if na.get('stats', {}).get(sprint_stat, 0) <= a.get('stats', {}).get(sprint_stat, 0):
            break
        a['keyboard']['keys'].append('key.keyboard.left.control')
    replay_action = self.actions[0]
    if replay_action.get('isGuiOpen', False):
        self.env.step(self.env.action_space.no_op())
        ac = self.env.action_space.no_op()
        ac['inventory' if replay_action.get('isGuiInventory') else 'use'] = 1
        self.env.step(ac)
        for _ in range(5):
            self.env.step(self.env.action_space.no_op())
        ma = replay_action['mouse']
        dx = (ma['x'] - 640) / 2
        dy = (ma['y'] - 360) / 2
        dx = ma.get('scaledX', dx)
        dy = ma.get('scaledY', dy)
        ac = self.env.action_space.no_op()
        ac['camera'] = mc.mouse_to_camera({'dx': dx, 'dy': dy})
        ob, _, _, _ = self.env.step(ac)
    return ob