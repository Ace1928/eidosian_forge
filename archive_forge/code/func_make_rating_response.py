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
def make_rating_response(self, rating):
    action = super().act()
    reply = str(action['text'])
    action['reward'] = rating
    last_message = self.history.history_strings[-1]
    if rating == 0:
        action['text'] = f'Okay, thanks! {CONTINUE} ("{last_message}"): {reply}'
    elif rating == 1:
        action['text'] = f'Great, thanks! {CONTINUE} ("{last_message}"): {reply}'
    return (action, reply)