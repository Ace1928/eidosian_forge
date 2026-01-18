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
def get_hyp_from_finished(self, hypothesis_tail):
    """
        Extract hypothesis ending with EOS at timestep with hyp_id.

        :param timestep:
            timestep with range up to len(self.outputs)-1
        :param hyp_id:
            id with range up to beam_size-1
        :return:
            hypothesis sequence
        """
    assert self.outputs[hypothesis_tail.timestep][hypothesis_tail.hypid] == self.eos
    assert hypothesis_tail.tokenid == self.eos
    hyp_idx = []
    endback = hypothesis_tail.hypid
    for i in range(hypothesis_tail.timestep, -1, -1):
        hyp_idx.append(self.HypothesisTail(timestep=i, hypid=endback, score=self.all_scores[i][endback], tokenid=self.outputs[i][endback]))
        endback = self.bookkeep[i - 1][endback]
    return hyp_idx