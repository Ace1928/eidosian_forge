import logging
import os
from pathlib import Path
from time import sleep
from typing import Callable, List, Optional, Union
import numpy as np
import tensorflow as tf
from huggingface_hub import Repository, create_repo
from packaging.version import parse
from . import IntervalStrategy, PreTrainedTokenizerBase
from .modelcard import TrainingSummary
from .modeling_tf_utils import keras
def generation_function(inputs, attention_mask):
    return self.model.generate(inputs, attention_mask=attention_mask, **self.generate_kwargs)