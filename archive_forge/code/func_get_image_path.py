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
def get_image_path(self, opt):
    """
        Return the path to the data directory and to the image directory.

        Is based on opt fields: task, datatype (train, valid, test), datapath.

        Subclass can override this.
        """
    data_path = self.get_data_path(opt)
    if opt.get('image_path', None):
        image_path = opt['image_path']
    else:
        image_path = os.path.join(data_path, 'images')
    return image_path