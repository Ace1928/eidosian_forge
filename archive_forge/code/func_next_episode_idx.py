import copy
from typing import List, Tuple, Optional, TypeVar
from parlai.core.agents import Agent, create_agent_from_shared
from parlai.core.image_featurizers import ImageLoader
from parlai.core.loader import load_teacher_module
from parlai.core.loader import register_teacher  # noqa: F401
from parlai.core.message import Message
from parlai.core.metrics import TeacherMetrics, aggregate_named_reports
from parlai.core.opt import Opt
from parlai.utils.conversations import Conversations
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import AttrDict, no_lock, str_to_msg, warn_once
from parlai.utils.distributed import get_rank, num_workers, is_distributed
import parlai.utils.logging as logging
from abc import ABC, abstractmethod
import concurrent.futures
from threading import Thread
import queue
import random
import time
import os
import torch
import json
import argparse
def next_episode_idx(self, num_eps=None, loop=None):
    """
        Return the next episode index.

        :param num_eps:
            default None uses ``num_episodes`` value.
        :param loop:
            default None loops during training but not evaluation.
        """
    if num_eps is None:
        num_eps = self.num_episodes()
    if loop is None:
        loop = self.training
    if self.random:
        new_idx = random.randrange(num_eps)
    else:
        with self._lock():
            self.index.value += 1
            if loop:
                self.index.value %= num_eps
            new_idx = self.index.value
    return new_idx