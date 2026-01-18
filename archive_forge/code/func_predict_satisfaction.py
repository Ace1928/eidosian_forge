import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import round_sigfigs, warn_once
from parlai.utils.torch import padded_tensor
from parlai.agents.transformer.transformer import TransformerRankerAgent
from .feedback_classifier.feedback_classifier import FeedbackClassifierRegex
from .modules import SelfFeedingModel
def predict_satisfaction(self, observation):
    if self.opt['regex']:
        prob = self.rating_classifier.predict_proba(observation['text_vec'])
    else:
        prob = self.model.score_satisfaction(observation['text_vec'].reshape(1, -1))
    return prob.item()