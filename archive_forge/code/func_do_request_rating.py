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
def do_request_rating(self, positivity):
    """
        Decide whether to request a rating this turn.
        """
    if not self.opt['request_rating']:
        return False
    elif len(self.history.history_strings) < 2:
        return False
    elif self.requested_rating:
        return False
    elif random.random() < self.opt['rating_frequency']:
        return True
    else:
        gap = abs(positivity - self.opt['rating_threshold'])
        return gap < self.opt['rating_gap']