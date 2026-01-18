from typing import Dict, Any
from abc import abstractmethod
from itertools import islice
import os
from tqdm import tqdm
import random
import torch
from parlai.core.opt import Opt
from parlai.utils.distributed import is_distributed
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import warn_once
from parlai.utils.torch import (
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.core.metrics import AverageMetric
import parlai.utils.logging as logging
def block_repeats(self, cand_preds):
    """
        Heuristic to block a model repeating a line from the history.
        """
    history_strings = []
    for h in self.history.history_raw_strings:
        history_strings.extend(h.split('\n'))
    new_preds = []
    for cp in cand_preds:
        np = []
        for c in cp:
            if c not in history_strings:
                np.append(c)
        new_preds.append(np)
    return new_preds