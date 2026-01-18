from collections import OrderedDict
import os
import torch
from torch.serialization import default_restore_location
from typing import Any, Dict, List
from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
def print_agent_act(self):
    """
        Print a sample act from the converted agent.
        """
    self.agent.observe({'text': "What's your favorite kind of ramen?", 'episode_done': False})
    print(self.agent.act())