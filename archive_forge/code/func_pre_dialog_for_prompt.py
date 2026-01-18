import random
import sys
import torch
import gc
from openchat.base.envs.base import BaseEnvironment
from openchat.base import (
from openchat.utils.terminal_utils import (
def pre_dialog_for_prompt(self, agent):
    user_name = cinput(f'[YOUR NAME]: ', color=self.special_color)
    bot_name = cinput(f"[{agent.name.upper()}'s NAME]: ", color=self.special_color)
    agent.name = bot_name
    cprint(f"\n[SYSTEM]: Please input story you want.\n[SYSTEM]: The story must contains '{user_name}' and '{bot_name}'.\n", color=self.system_color)
    story = cinput('[STORY]: ', color=self.special_color)
    while user_name not in story or bot_name not in story:
        cprint(f"\n[SYSTEM]: Please input story you want.\n[SYSTEM]: The story MUST contains '{user_name}' and '{bot_name}'.\n", color=self.system_color)
        story = cinput('[STORY]: ', color=self.special_color)
    cprint(f'[STORY]: Story setting complete.\n', color=self.special_color)
    story += f' {user_name} and {bot_name} start talking. '
    story += f'{user_name}: Hello {bot_name}. '
    story += f'{bot_name}: Hi {user_name}. '
    agent.add_prompt(self.histories, self.user_id, story)
    return (user_name, bot_name)