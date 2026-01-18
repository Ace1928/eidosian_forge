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
def get_rescored_finished(self, n_best=None):
    """
        Return finished hypotheses in rescored order.

        :param n_best:
            how many n best hypothesis to return
        :return:
            list with hypothesis
        """
    rescored_finished = []
    for finished_item in self.finished:
        current_length = finished_item.timestep + 1
        length_penalty = math.pow((1 + current_length) / 6, 0.65)
        rescored_finished.append(self.HypothesisTail(timestep=finished_item.timestep, hypid=finished_item.hypid, score=finished_item.score / length_penalty, tokenid=finished_item.tokenid))
    srted = sorted(rescored_finished, key=attrgetter('score'), reverse=True)
    if n_best is not None:
        srted = srted[:n_best]
    return srted