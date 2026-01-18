import logging
import os
from threading import Event
from typing import Dict, List, Optional, Union
import torch
from .async_schedule import AsyncEventLoop, ModuleWrapper
from .messages import MakeTransport
from .microbatch import Batch
from .skip.layout import SkipLayout
from .skip.tracker import SkipTrackerThroughPotals
@property
def checkpoint_stop(self) -> int:
    training = self.partitions[0].module.training
    if not training:
        return 0
    return self.__checkpoint_stop