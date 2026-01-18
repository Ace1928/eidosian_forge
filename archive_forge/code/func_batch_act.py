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
def batch_act(self, observations):
    """
        Act on a batch of observations.

        :param observations:
            list of observations

        :return:
            A list of acts, one for each observation
        """
    is_training = any(['labels' in obs for obs in observations])
    valid_obs, valid_indexes = self.filter_valid_obs(observations, is_training)
    image_feats = self.extract_image_feats(valid_obs)
    personalities = [v.get('text', '') for v in valid_obs]
    chosen_captions = None
    med_rank = None
    if is_training:
        loss, num_correct, num_examples = self.train_step(valid_obs, image_feats, personalities)
    else:
        loss, num_correct, num_examples, med_rank, chosen_captions = self.eval_step(valid_obs, image_feats, personalities)
    self.update_metrics(loss, num_correct, num_examples, med_rank)
    result = [{'text': 'No Response During Training'} for _ in range(len(observations))]
    if chosen_captions is not None:
        for i, index_obs in enumerate(valid_indexes):
            result[index_obs]['text'] = chosen_captions[i][0]
            result[index_obs]['text_candidates'] = chosen_captions[i]
    return result