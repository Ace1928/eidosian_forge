import random
import sys
import torch
import gc
from openchat.base.envs.base import BaseEnvironment
from openchat.base import (
from openchat.utils.terminal_utils import (
def pre_dialog_for_wow(self, agent):
    cprint(f"[SYSTEM]: Please input topic for Wizard of wikipedia.\n[SYSTEM]: Enter '.topic' if you want to check random topic examples.\n", color=self.system_color)
    while True:
        _topic = cinput('[TOPIC]: ', color=self.special_color)
        if _topic == '.topic':
            random_list = agent.topic_list
            random.shuffle(random_list)
            random_list = random_list[:4]
            _topic = cprint(f'[TOPIC]: {random_list}\n', color=self.special_color)
        elif _topic in agent.topic_list:
            cprint(f'[TOPIC]: Topic setting complete.\n', color=self.special_color)
            agent.set_topic(_topic)
            break
        else:
            _topic = cprint(f'[TOPIC]: Wrong topic: {_topic}. Please enter validate topic.\n', color=self.special_color)