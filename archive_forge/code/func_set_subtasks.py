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
def set_subtasks(self, opt):
    if opt.get('subtasks', None):
        if isinstance(opt['subtasks'], list):
            subtasks = opt['subtasks']
        else:
            subtasks = opt['subtasks'].split(',')
    else:
        subtasks = [task.split(':')[1] for task in opt['task'].split(',')]
    if subtasks[0] == 'diafee':
        subtasks = ['dialog', 'feedback']
    elif subtasks[0] == 'diasat':
        subtasks = ['dialog', 'satisfaction']
    elif subtasks[0] == 'all':
        subtasks = ['dialog', 'feedback', 'satisfaction']
    self.subtasks = subtasks
    opt['subtasks'] = subtasks