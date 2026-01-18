from typing import Dict, Any, Union, List, Tuple, Optional
from abc import ABC, abstractmethod
import random
import os
import torch
import parlai.utils.logging as logging
from torch import optim
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.nn.lr_scheduler import ParlAILRScheduler
from parlai.core.message import Message
from parlai.utils.distributed import is_distributed
from parlai.utils.misc import AttrDict, warn_once
from parlai.utils.fp16 import (
from parlai.core.metrics import (
from parlai.utils.distributed import is_primary_worker
from parlai.utils.torch import argsort, compute_grad_norm, padded_tensor, atomic_save
def record_local_metric(self, keyname: str, values: List[Metric]):
    """
        Record an example-level metric for all items in the batch.

        Local metrics are maybe recorded anywhere within batch act. They will
        automatically be collated and returned at the end of batch_act. The
        beginning of batch_act resets these, so you may not use them during
        observe.

        Example local metrics include ppl, token_acc, any other agent-specific
        metrics.
        """
    if not self.__local_metrics_enabled:
        return
    if keyname in self._local_metrics:
        raise KeyError(f'Already recorded metrics for {keyname}')
    self._local_metrics[keyname] = values