from parlai.core.torch_agent import TorchAgent, Output, Batch
from parlai.utils.misc import round_sigfigs
from parlai.utils.torch import padded_tensor, argsort, neginf
from .modules import Seq2seq, opt_to_kwargs
from .util import ConvAI2History, show_beam_cands, reorder_extrep2gram_qn
from .controls import (
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, Counter
from operator import attrgetter
import os
import math
import json
import tempfile
import copy
@staticmethod
def get_pretty_hypothesis(list_of_hypotails):
    """
        Return prettier version of the hypotheses.
        """
    hypothesis = []
    for i in list_of_hypotails:
        hypothesis.append(i.tokenid)
    hypothesis = torch.stack(list(reversed(hypothesis)))
    return hypothesis