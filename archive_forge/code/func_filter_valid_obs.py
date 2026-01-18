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
def filter_valid_obs(self, observations, is_training):
    """
        Filter out invalid observations.
        """
    label_key = 'labels' if is_training else 'eval_labels'
    valid_obs = []
    valid_indexes = []
    seen_texts = set()
    for i in range(len(observations)):
        if 'image' in observations[i]:
            if self.fixed_cands is not None:
                valid_obs.append(observations[i])
                valid_indexes.append(i)
            else:
                text = observations[i][label_key][0]
                if text not in seen_texts:
                    seen_texts.add(text)
                    valid_obs.append(observations[i])
                    valid_indexes.append(i)
    return (valid_obs, valid_indexes)