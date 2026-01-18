from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.utils.misc import round_sigfigs
from .modules import TransresnetModel
from parlai.tasks.personality_captions.build import build
import os
import random
import json
import numpy as np
import torch
import tqdm
def load_personalities(self):
    """
        Load and return the list of personalities.
        """
    personality_path = os.path.join(self.opt['datapath'], 'personality_captions/personalities.txt')
    if 'yfcc_path' not in self.opt:
        self.opt['yfcc_path'] = 'temp_path'
    build(self.opt)
    del self.opt['yfcc_path']
    perss = []
    with open(personality_path) as f:
        for line in f:
            if 'Trait' not in line:
                perss.append(line[0:-1])
    return perss