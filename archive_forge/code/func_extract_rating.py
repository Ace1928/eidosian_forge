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
def extract_rating(self):
    """
        Convert user response to rating request from text to an integer rating.
        """
    if self.last_rating == 'positive':
        return 1
    elif self.last_rating == 'negative':
        return -1
    else:
        return 0