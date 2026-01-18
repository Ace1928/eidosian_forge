import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm.auto import tqdm
from .trainer_utils import IntervalStrategy, has_length
from .training_args import TrainingArguments
from .utils import logging
@property
def callback_list(self):
    return '\n'.join((cb.__class__.__name__ for cb in self.callbacks))