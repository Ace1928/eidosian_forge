import random
import sys
import torch
import gc
from openchat.base.envs.base import BaseEnvironment
from openchat.base import (
from openchat.utils.terminal_utils import (
def pre_dialog_for_special_tasks(self, agent):
    if isinstance(agent, ConvAI2Agent):
        return self.pre_dialog_for_convai2(agent)
    if isinstance(agent, WizardOfWikipediaAgent):
        return self.pre_dialog_for_wow(agent)
    if isinstance(agent, PromptAgent):
        return self.pre_dialog_for_prompt(agent)