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
def get_task_candidates_path(self):
    path = self.opt['model_file'] + '.cands-' + self.opt['task'] + '.cands'
    if os.path.isfile(path) and self.opt['fixed_candidate_vecs'] == 'reuse':
        return path
    logging.warn(f'Building candidates file as they do not exist: {path}')
    from parlai.scripts.build_candidates import build_cands
    from copy import deepcopy
    opt = deepcopy(self.opt)
    opt['outfile'] = path
    opt['datatype'] = 'train:evalmode'
    opt['interactive_task'] = False
    opt['batchsize'] = 1
    build_cands(opt)
    return path