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
def check_finished(self):
    """
        Check if self.finished is empty and add hyptail in that case.

        This will be suboptimal hypothesis since the model did not get any EOS
        """
    if len(self.finished) == 0:
        self.outputs[-1][0] = self.eos
        hyptail = self.HypothesisTail(timestep=len(self.outputs) - 1, hypid=0, score=self.all_scores[-1][0], tokenid=self.outputs[-1][0])
        self.finished.append(hyptail)