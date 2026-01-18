from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...generation.configuration_utils import GenerationConfig
from ...generation.tf_logits_process import TFLogitsProcessorList
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
def encoder(self):
    return self.model.encoder